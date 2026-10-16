"""Export all frozen confirmation scores and localize follow-up corrections."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_review_utility_v2 import verify_receipts
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_window_optimization import read_json
from write_landslide_dual_reference_report import full_table

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs
from gnss_sim.landslide_review_utility import correction_utility


def draw_record(data, cid, references, predictions, out):
    case = read_json(data / "inputs" / cid / "input.json")
    values = np.array([[np.nan] * 3 if row is None else row for row in case["displacement_mm"]])
    observed = np.isfinite(values).all(axis=1)
    states = {"Author-AI reference": [row["activity_label"] for row in references["primary_ai"][cid]],
              "Latent process": [row["activity_label"] for row in references["secondary_process"][cid]]}
    for rep in (0, 1):
        for label, method in (("Direct", f"direct_visual_r{rep}"),
                              ("Fixed", f"fixed_image_review_r{rep}__whole_target"),
                              ("Original", f"adaptive_no_feedback_r{rep}__conflict_scope"),
                              ("First only", f"first_only_r{rep}"),
                              ("Censored", f"followup_censored_r{rep}"),
                              ("Wide", f"followup_wide_r{rep}")):
            states[f"{label} r{rep}"] = predictions[method][cid]["activity"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True,
                            gridspec_kw={"height_ratios": [2, 2, 2, 5]})
    for component, name in enumerate(("N", "E", "U")):
        axes[component].plot(values[:, component], linewidth=.7)
        axes[component].set_ylabel(name + " mm")
        axes[component].grid(alpha=.2)
    codes = np.array(list(states.values())) + 1
    codes[:, ~observed] = 3
    axes[3].imshow(codes, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                   cmap=ListedColormap(["#edc46a", "#e0e7e0", "#407ea0", "#777777"]),
                   extent=(-.5, len(values) - .5, len(states) - .5, -.5))
    axes[3].set_yticks(range(len(states)), list(states), fontsize=8)
    axes[3].set_xlim(0, len(values) - 1)
    axes[3].set_xlabel("Original day; blue=activity, pale=stationary, amber=unknown, gray=missing")
    fig.suptitle(cid + " | post-score author-AI inspection; frozen labels and predictions unchanged")
    fig.tight_layout()
    fig.savefig(out / (cid + ".png"), dpi=140)
    plt.close(fig)


def export(comparison, analysis, out):
    verify_manifest(analysis)
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    run = comparison / "run"
    data = Path(read_json(comparison / "source-bindings.json")["data"])
    receipts = verify_receipts(run)
    schedules = read_json(run / "followup-schedules.json")
    observed = {cid: np.array([row is not None for row in read_json(
        data / "inputs" / cid / "input.json")["displacement_mm"]])
        for cid in read_json(run / "protocol.json")["case_ids"]}
    references, tables, predictions = {}, {}, {}
    corrections, changes, bootstrap = [], [], []
    rng = np.random.default_rng(2026101193)
    for target in ("primary_ai", "secondary_process"):
        references[target] = read_json(analysis / target / "reference-labels.json")
        predictions[target] = read_json(analysis / target / "predictions.json")
        with (analysis / target / "summary.csv").open(encoding="utf-8-sig") as stream:
            tables[target] = list(csv.DictReader(stream))
        with (analysis / target / "per-record.csv").open(encoding="utf-8-sig") as stream:
            records = list(csv.DictReader(stream))
        cids = sorted(references[target])
        indices = rng.integers(0, len(cids), (2000, len(cids)))
        for rep in (0, 1):
            main = f"followup_censored_r{rep}"
            for control in (f"adaptive_no_feedback_r{rep}__conflict_scope",
                            f"followup_fixed_short_r{rep}", f"first_only_r{rep}",
                            f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target"):
                for cid, rows in references[target].items():
                    actual = np.array([row["activity_label"] for row in rows])
                    before = np.array(predictions[target][control][cid]["activity"])
                    after = np.array(predictions[target][main][cid]["activity"])
                    corrections.append({"target": target, "case_id": cid, "repeat": rep,
                        "control": control, "treatment": main,
                        **correction_utility(actual, before, after, observed[cid]),
                        "tp_difference": int(((actual == 1) & (after == 1)).sum()
                                             - ((actual == 1) & (before == 1)).sum()),
                        "fp_difference": int(((actual == 0) & (after == 1)).sum()
                                             - ((actual == 0) & (before == 1)).sum())})
                    for old in (-1, 0, 1):
                        for new in (-1, 0, 1):
                            if old == new:
                                continue
                            for start, stop in runs(observed[cid] & (before == old) & (after == new)):
                                changes.append({"target": target, "case_id": cid, "repeat": rep,
                                    "control": control, "old": old, "new": new, "start": start,
                                    "stop": stop, "reference_activity": actual[start:stop].tolist()})
                if control.startswith(("adaptive_no_feedback", "followup_fixed_short", "first_only")):
                    values = {}
                    for method in (main, control):
                        mapping = {row["case_id"]: row for row in records if row["method"] == method}
                        counts = np.array([[int(mapping[cid][key]) for key in ("tp", "fp", "fn")] for cid in cids])
                        total = counts[indices].sum(axis=1)
                        values[method] = 2 * total[:, 0] / np.maximum(1, 2 * total[:, 0] + total[:, 1] + total[:, 2])
                    bootstrap.append({"target": target, "repeat": rep, "control": control,
                        "descriptive_95_percentile_interval": np.quantile(
                            values[main] - values[control], [.025, .975]).tolist(),
                        "scope": "2000 whole-record resamples; post-score descriptive only; repeats not independent"})
    write_csv(out / "per-record-corrections.csv", corrections)
    write_json(out / "changed-intervals.json", changes)
    write_json(out / "bootstrap.json", bootstrap)
    started = list((run / "requests").glob("*.started.json"))
    cost = {"launches": len(started), "receipts": len(receipts),
            "reported_tokens": sum(value["usage"]["total_tokens"] for value in receipts.values()),
            "seconds": read_json(run / "state.json")["seconds"],
            "failed_responses": sum(value["status"] != "returned_json" for value in receipts.values()),
            "parse_or_transport_failures": read_json(run / "failures.json"), "retries": 0,
            "scope": "Shared actual inference; per-method deployment costs are in summary.csv"}
    trigger = [key for key, value in schedules.items() if value["budget"] == 2
               and value["continuation"]["numeric_admitted"]]
    counts = {}
    for target, reference in references.items():
        actual = np.concatenate([np.array([row["activity_label"] for row in rows]) for rows in reference.values()])
        counts[target] = {"activity": int((actual == 1).sum()), "stationary": int((actual == 0).sum()),
                          "unknown_including_missing": int((actual == -1).sum())}
    write_json(out / "evidence-summary.json", {"cost": cost, "reference_counts": counts,
        "triggered": trigger, "record_repeat_pairs": len(schedules),
        "two_review_pairs": sum(row["budget"] == 2 for row in schedules.values()),
        "candidate_status": "STOPPED", "default_changed": False,
        "decision_sha256": sha(analysis / "decision.json")})
    for cid in ("case_0159", "case_0160", "case_0164", "case_0166"):
        draw_record(data, cid, references, predictions["primary_ai"], out)
    # Non-overlapping raw medians help interpret the failed local view; they never alter labels.
    case = read_json(data / "inputs/case_0159/input.json")
    values = np.array([[np.nan] * 3 if row is None else row for row in case["displacement_mm"]])
    blocks = []
    for start, stop in ((540, 581), (581, 614), (614, 647), (647, 679), (679, 720)):
        blocks.append({"start": start, "stop": stop, "observed_days": int(np.isfinite(values[start:stop]).all(axis=1).sum()),
                       "median_NEU_mm": np.nanmedian(values[start:stop], axis=0).tolist()})
    write_json(out / "case_0159-raw-block-medians.json", blocks)
    text = "# 顺序第二观察的冻结确认：候选停止\n\n## Material Passport\n\n"
    text += "- Origin Skill: academic-research-suite / experiment-agent\n- Origin Mode: run / validate\n"
    text += "- Origin Date: 2026-10-11\n- Verification Status: 全部预测冻定后评分，逐回执验证通过；离线评分重放另存\n"
    text += "- Version Label: sequential-followup-confirmation-v1\n\n"
    text += "12条未见受控记录，两遍、46方法×2目标。新候选按生成前登记条件失败，状态STOPPED；不选择更好消融或重复替换主方案，不再重抽确认批次救分。\n\n"
    text += "主观测F1 .833273/.784595，原方法及同触发固定短窗 .833273/.817027，仅第一窗 .834783/.799625。主对原平均差−.016216、弱R平均差−.077778、净确定纠错−84；对仅第一窗平均差−.008270、净纠错−23。过程主 .769464/.794320，对原平均差−.013234、净−98；对第一窗+.010135、净+71，但不能覆盖主目标和原方法比较失败。\n\n"
    text += "生成前冻结seed2026101191、排列2026101192、stream156—167；旧六家族各2，不重抽。AI观测和过程模型使用原共同6/6训练验证与原目标/分布，生成前冻结，未调参。作者AI先实际看24原始图、冻结活动，再看12辅助图并冻结阶段；标签不依赖本批预测/潜在标签，但作者参与开发且见过旧过程标签，不是独立专家或现场金标准。未知、缺测、边界和A/S/D掩码保留。\n\n"
    text += "第一回答之后保存第二目标再发请求；同记录/遍的相同问题共享当前批回执，未使用历史回答。原0/1/2预算、门槛、图/提示、49日权限和严格多数均不变。115回执验证、144方案逐记录重建、45个第一回答早于第二请求检查通过。主只在0159第二遍触发，1/24组合（1/9两次预算组合）；不能据结果扩大触发。\n\n"
    for target, title in (("primary_ai", "作者AI确定观测日期"), ("secondary_process", "全观测潜在过程")):
        text += f"## {title}\n\n" + full_table(tables[target]) + "\n\n"
    text += "## 全部纠错、阶段与成本\n\n"
    text += "两目标逐记录240组纠错、全部变更区间、未知与覆盖、A/S/D每类分数、事件IoU.3/.5和起止MAE均保存。描述bootstrap不进入冻结决策、不将两遍作为24个独立样本；零差重复不等于没有误差。两个目标的训练模型和参考都不同，不能将差异归因于单个因素。\n\n"
    text += f"实际共享{cost['launches']}启动/{cost['receipts']}回执、{cost['reported_tokens']:,}报告token、{cost['seconds']:.3f}秒，无推理失败/重试。主部署22/23调用、70,418/74,576token；第一窗18/18调用、54,123/54,190token；直接12/12、固定36/36。成本不包含AI参考制作、训练和全部特征预处理。\n\n"
    text += "统计解释边界：不报告显著性、不将小样本描述区间视为功效证明、不做因果外推；保持参考未知及选择时间、不挑重复/目标/消融、不按看过结果改阈值、不将计算调用数量当准确率，不把共同数值阶段头称为原生VLM阶段创新。\n\n"
    text += "## 复现\n\n评分源见scripts/analyze_landslide_sequential_followup.py；完整真实回答保存在comparison-v1/run/requests。离线评分到新目录，禁止重启顺序executor；逐回执重建见verification-v1。四幅原始及预测对齐图与数值块中位数保存在本report目录，实际视觉解读另存，不回写标签或预测。\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()}})
    print({"cost": cost, "triggers": trigger, "corrections": len(corrections), "counts": counts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.comparison.resolve(), args.analysis.resolve(), args.out.resolve())
