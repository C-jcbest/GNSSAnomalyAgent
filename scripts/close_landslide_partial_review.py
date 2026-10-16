"""Explain the stopped partial-update probe and bind its manuscript evidence."""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_partial_review import BASE, BATCHES, ROOT
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs
from gnss_sim.landslide_review_utility import correction_utility

RUN = BASE / "partial-review-development-v1"
ANALYSIS = RUN / "analysis"
OUT = BASE / "partial-review-development-report-v1"


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def draw(cid, labels, predictions, data):
    case = read_json(data / "inputs" / cid / "input.json")
    values = np.array([[np.nan] * 3 if row is None else row for row in case["displacement_mm"]])
    states = {"AI reference": [row["activity_label"] for row in labels[cid]]}
    for label, name in (("Global r1", "direct_visual_r1"),
                        ("Original r1", "adaptive_no_feedback_r1__conflict_scope"),
                        ("Partial update r1", "partial_adaptive_no_feedback_r1__conflict_scope"),
                        ("First partial r1", "partial_first_only_r1__conflict_scope"),
                        ("Fixed partial r1", "partial_fixed_image_review_r1__whole_target")):
        states[label] = predictions[name][cid]["activity"]
    figure, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                               gridspec_kw={"height_ratios": [2, 2, 2, 3]})
    for component, name in enumerate(("N", "E", "U")):
        axes[component].plot(values[:, component], linewidth=.7)
        axes[component].set_ylabel(name + " mm")
        axes[component].grid(alpha=.2)
    codes = np.array(list(states.values())) + 1
    codes[:, ~np.isfinite(values).all(axis=1)] = 3
    axes[3].imshow(codes, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
        cmap=ListedColormap(["#edc46a", "#e0e7e0", "#407ea0", "#777777"]),
        extent=(-.5, len(values) - .5, len(states) - .5, -.5))
    axes[3].set_yticks(range(len(states)), list(states), fontsize=8)
    axes[3].set_xlabel("Original day; blue=activity, pale=stationary, amber=unknown, gray=missing")
    axes[3].set_xlim(0, len(values) - 1)
    figure.suptitle(cid + " | exposed development; partial-update tradeoff, labels unchanged")
    figure.tight_layout()
    figure.savefig(OUT / (cid + ".png"), dpi=140)
    plt.close(figure)


def export():
    OUT.mkdir(exist_ok=False)
    (OUT / "source.py").write_bytes(Path(__file__).read_bytes())
    matches = {}
    replay = BASE / "partial-review-development-replay-v1/analysis"
    for path in ANALYSIS.rglob("*"):
        if path.is_file():
            relative = path.relative_to(ANALYSIS).as_posix()
            if path.read_bytes() != (replay / relative).read_bytes():
                raise ValueError("Offline score differs: " + relative)
            matches[relative] = sha(path)
    write_json(OUT / "replay-check.json", {"status": "passed", "exact_files": matches,
                                           "new_remote_calls": 0})
    baseline_checks, corrections, changed_intervals, tables = [], [], [], {}
    for batch, (name, process_analysis) in BATCHES.items():
        comparison = BASE / name
        data = Path(read_json(comparison / "source-bindings.json")["data"])
        if process_analysis is None:
            from analyze_landslide_codex_observation_reference import scoring_inputs
            labels = scoring_inputs(data / "codex-review")[0]
            targets = [(batch, labels, comparison / "analysis")]
        else:
            parent = BASE / process_analysis
            targets = [(batch, read_json(parent / "primary_ai/reference-labels.json"), parent / "primary_ai"),
                (batch.replace("ai_", "process_"), read_json(parent / "secondary_process/reference-labels.json"),
                 parent / "secondary_process")]
        observed = {cid: np.array([row is not None for row in read_json(
            data / "inputs" / cid / "input.json")["displacement_mm"]]) for cid in targets[0][1]}
        for target, reference, original in targets:
            predictions = read_json(RUN / batch / ("process-predictions.json" if target.startswith("process")
                                                   else "predictions.json"))
            table = {row["method"]: row for row in rows(ANALYSIS / target / "summary.csv")}
            tables[target] = table
            for previous in rows(original / "summary.csv"):
                method = previous["method"]
                for key in ("precision", "recall", "f1", "tp", "fp", "fn", "weak_recall"):
                    if key in previous and previous[key]:
                        if float(previous[key]) != float(table[method][key]):
                            raise ValueError("Historical baseline score changed: " + target + " " + method + " " + key)
                baseline_checks.append({"target": target, "method": method, "status": "exact_common_metrics"})
            for rep in (0, 1):
                main = f"partial_adaptive_no_feedback_r{rep}__conflict_scope"
                for control in (f"adaptive_no_feedback_r{rep}__conflict_scope",
                                f"partial_fixed_image_review_r{rep}__whole_target",
                                f"partial_first_only_r{rep}__conflict_scope"):
                    for cid, records in reference.items():
                        actual = np.array([row["activity_label"] for row in records])
                        before = np.array(predictions[control][cid]["activity"])
                        after = np.array(predictions[main][cid]["activity"])
                        changed = observed[cid] & (before != after)
                        corrections.append({"target": target, "repeat": rep, "control": control, "case_id": cid,
                            **correction_utility(actual, before, after, observed[cid]),
                            "tp_difference": int(((actual == 1) & (after == 1)).sum()
                                                 - ((actual == 1) & (before == 1)).sum()),
                            "fp_difference": int(((actual == 0) & (after == 1)).sum()
                                                 - ((actual == 0) & (before == 1)).sum()),
                            "unknown_reference_changed": int((changed & (actual == -1)).sum())})
                        for start, stop in runs(changed):
                            changed_intervals.append({"target": target, "repeat": rep, "control": control,
                                "case_id": cid, "start": start, "stop": stop,
                                "actual": actual[start:stop].tolist(), "before": before[start:stop].tolist(),
                                "after": after[start:stop].tolist()})
            if target == "ai_0157_0168":
                for cid in ("case_0161", "case_0163"):
                    draw(cid, reference, predictions, data)
    write_json(OUT / "baseline-check.json", {"status": "passed", "comparisons": baseline_checks})
    write_csv(OUT / "per-record-corrections.csv", corrections)
    write_json(OUT / "changed-intervals.json", changed_intervals)
    compact = ["| 目标 | 原方法F1 r0/r1 | 部分更新F1 r0/r1 | 仅第一窗部分更新F1 r0/r1 | 部分更新事件F1 .3 r0/r1 |",
               "| --- | --- | --- | --- | --- |"]
    for target, table in tables.items():
        values = []
        for stem in ("adaptive_no_feedback", "partial_adaptive_no_feedback", "partial_first_only"):
            values.append("/".join(f"{float(table[f'{stem}_r{rep}__conflict_scope']['f1']):.4f}" for rep in (0, 1)))
        values.append("/".join(f"{float(table[f'partial_adaptive_no_feedback_r{rep}__conflict_scope']['event_f1_iou03']):.4f}"
                              for rep in (0, 1)))
        compact.append("| " + " | ".join([target] + values) + " |")
    (OUT / "compact-table.md").write_text("\n".join(compact), encoding="utf-8")
    write_json(OUT / "delivery-summary.json", {"candidate": "STOPPED", "methods_by_target": {
        target: len(table) for target, table in tables.items()}, "exact_replay_files": len(matches),
        "baseline_comparisons": len(baseline_checks), "correction_rows": len(corrections),
        "new_remote_calls": 0, "new_records": 0, "new_training": False, "default_changed": False})
    print(read_json(OUT / "delivery-summary.json"))


def manuscript():
    out = BASE / "manuscript-v35"
    out.mkdir(exist_ok=False)
    path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    old_bytes = path.read_bytes()
    (out / "manuscript-v34-original.md").write_bytes(old_bytes)
    old = old_bytes.decode("utf-8").replace("\r\n", "\n")
    table = (OUT / "compact-table.md").read_text(encoding="utf-8")
    section = r"""### 6.17 成功弃判的部分更新语义：覆盖改善不能替代活动效果

为检验局部未知是否必须撤销全局判断，新增一个无新门槛的聚合候选。设全局判断为 $g_t$，原严格多数局部结果为 $l_t$，$U_t=1$ 表示当日有局部覆盖、全部覆盖复核成功且全部回答未知，则

$$
h_t=\begin{cases}g_t,&U_t=1,\\l_t,&U_t=0.\end{cases}
$$

仍只在原49日权限内修订，其余保留旧值。明确正/负回答仍按原多数处理；确定冲突、失败和缺测不由旧值补齐，未知初判不产生确定结果。它是保留旧判断的惯性策略，不是新的GNSS正证据或校准不确定性。与已停止“保留全部旧阳性”不同，本候选允许明确局部阴性撤回活动。数值只决定原180日观察及权限，没有恢复已停止的端点98日模块或双反证保护。

使用三批完整已暴露AI记录0133—0168，共36条，以及后两批全观测过程目标。先冻结聚合源码、全量候选预测、原始回执绑定和继续条件，再评分，但数据和历史结果已经暴露，只能称开发。原图、提示、两次预算、数值模型和阶段头均未改，358个历史回执复用，0新远端请求/训练/样本。原34方法、固定和同次数位置的相同聚合、整目标权限、保留全部旧阳性、便宜第一窗均报告；三旧目标54方法、两最新目标64方法，共290方法/目标结果，未挑错误子集。

""" + table + """

生成评分前规定36条合并日计数的两遍平均F1相对原、同语义固定及同语义第一窗均大于0、净确定纠错均为正，且对原弱召回不降。实际对原平均F1−.007476、弱R差0、净纠错+36；对固定+.026000、净+94；对第一窗−.009664、净−76。原比较和第一窗比较失败，候选STOPPED，不根据净纠错或覆盖挑指标救分，不再给未知恢复添加新门槛。

前两AI批次日F1无变化，但0152两遍各75个确认活动日从未知变确定静稳，没有活动TP增量或净正确增量。最新批次第二遍在0161恢复161个确定静稳区的旧活动误报，TP不变，F1 .8170降.7773；其活动原本被局部成功未知抑制。0163每遍恢复18个正确静稳判断，另100个确认活动日从未知改静稳，活动F1不变但错误性质改变：弃判变成确定漏检。合计350个确认活动日的弃判转为确定静稳。最新第二遍主覆盖.9598升.9979、事件F1 .7692降.6400；仅覆盖或净确定纠错改善不足以建立识别贡献。净正确计数中，未知和错误确定判断都不是正确，未知转错误不会扣额外正确分，因此必须同时看FP和确定漏检。过程最新第二遍增3TP却增167FP，F1 .8208降.7920，弱召回不变。未知参考仍未补齐或赚静稳分；全部逐日变化与损失保留。

实际原始及结果对齐图的视觉核查、数值对齐、360逐记录纠错和194个历史基线共用指标重现另存。五业务测试与Ruff通过，评分18文件字节精确重放；不将上述核验当方法有效或独立标签可信的证明。累计成本仍为3118启动/3117回执/至少13610633token，受控168/半合成60不变。主方法仍为原冲突位置候选，本更新未接默认。

本轮官方arXiv定向检索保存16条元数据，实际读取其中6条摘要。进一步实读Harness Compilation[14]第3节开头、第4节选择及边际价值段和附录D.4的弃判评分取舍：工具是否有用取决于可识别的子集，保留/移除决定不能只按单一平均指标。它使用教师与独立验证选择外部控制，任务为VQA，不是GNSS活动识别。实读Visual Abstention[15]第5.1节的可行/不可行成对监督、先判断再绘图训练；其拒绝是图像编辑请求不可行，与本任务的观测不可辨不同，不能套用其发布拒绝率证明本任务安全。未确认这两篇作者源码/权重，也未执行作者模型；TimeEvo公开源码的既有阅读范围未扩大。一般弃判或外部决策分配不是本文首创。

[完整290组评分、消融、逐记录纠错与停止结果](landslide-partial-review-development-results.md)。接下来的方法开发须改善实际运动证据消费而非简单恢复旧判断；先用完整开发记录验证识别效果，不恢复已停止保护或弃判回填。

"""
    text = old.replace("完整中文初稿v34", "完整中文初稿v35", 1)
    text = text.replace("## 7 讨论", section + "## 7 讨论", 1)
    text = text.replace("## 9 结论\n\n", "## 9 结论\n\n成功局部弃判保留全局判断也未形成稳定收益：覆盖与净确定纠错增加，活动F1却下降、平台误报恢复、部分弃判变成确定漏检。该聚合候选停止，研究不能以更多确定输出替代活动识别改善。\n\n", 1)
    text = text.replace("## 补充结果与完整数据", "[14] Fan, M., Liu, Y., Zhao, J., Teng, Z., Chen, S., & Liu, W. (2026). Harness Compilation: Which Decisions Should a Small Vision-Language Model Keep? [arXiv:2610.11231v1](https://arxiv.org/html/2610.11231v1). 已读第3节开头、第4节相关选择/价值段与附录D.4的弃判取舍；未确认作者源码，未复现。\n\n[15] Shi, C., Yang, C., Yang, T., White, I., Chen, Y., Berg-Kirkpatrick, T., & Ma, X. (2026). Visual Abstention in Unified Multimodal Models. [arXiv:2610.07887v1](https://arxiv.org/html/2610.07887v1). 已读摘要及第5.1节成对可行性训练；图像编辑任务，未执行作者模型或确认作者源码/权重。\n\n## 补充结果与完整数据", 1)
    formulas = re.compile(r"\$\$[\s\S]*?\$\$|(?<![\\$])\$(?!\$)[^$\n]+\$(?!\$)").findall(old)
    assert all(value in text for value in formulas)
    assert all(line in text for line in old.splitlines() if line.startswith("|"))
    path.write_text(text, encoding="utf-8", newline="\n")
    write_json(out / "content-check.json", {"old_formulas_preserved": len(formulas),
        "source_sha256": sha(path), "prior_backup_sha256": sha(out / "manuscript-v34-original.md")})
    report = (ANALYSIS / "report.md").read_text(encoding="utf-8")
    report = "# 成功弃判的部分更新开发：候选停止\n\n" + section + "\n\n" + report
    (ROOT / "docs/landslide-partial-review-development-results.md").write_text(report, encoding="utf-8")
    status = """## 2026-10-11：部分更新语义开发失败，候选停止，初稿v35

[完整结果](landslide-partial-review-development-results.md)：全36条已暴露作者AI记录及两过程目标，先冻结成功全部弃判时保留旧判断的新聚合，明确相反局部判断仍可改写，失败与缺测不恢复。三目标54方法、两最新目标64，共290结果。主对原平均F1−.007476、弱R差0、净纠错+36；对第一窗−.009664、净−76，按条件STOPPED，不接默认，不加门槛救分。最新第二遍恢复161个旧FP，另100活动日从弃判变确定静稳；覆盖改善不是活动贡献。五业务测试/Ruff、194基线共用指标重现、18文件字节精确重放及360组逐记录纠错保留。

358历史回执复用，0新请求/训练/记录；累计3118启动/3117回执/至少13610633token、受控168/半合成60不变。[初稿v35](landslide-adaptive-review-manuscript-draft.md)保留LaTeX、旧表与v34原字节。检索16条近期官方元数据，实读6摘要、HC决策/弃判评分段及VisTA成对可行性训练段；未确认两篇作者源码，不转用发布效果。作者AI身份不变，goal active。

下一步优先改善VLM消费真实运动证据的机制，先在完整已暴露开发记录小试；不恢复已停止用途选择、端点第二观察、反证保护或弃判回填，不围绕本批救参数。暂无领域复核者，按用户授权继续原始视觉＋数值辅助自主研究。

下方为历史记录。

"""
    for name in ("project-status.md", "landslide-adaptive-review-method.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / name
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + status + rest.lstrip("\n"), encoding="utf-8", newline="\n")
    write_json(out / "run-ledger-end.json", {"goal_status": "active", "turn_classification": "progress",
        "candidate": "STOPPED", "new_remote_calls": 0, "new_records": 0, "new_training": False,
        "cumulative_launches": 3118, "cumulative_receipts": 3117, "at_least_reported_tokens": 13610633,
        "controlled_records": 168, "semisynthetic_records": 60,
        "next_work": "Improve actual motion-evidence consumption using complete exposed development; do not rescue partial-update policy"})


if __name__ == "__main__":
    export()
    manuscript()
