"""Render the stopped development probe and trace changes to frozen raw inputs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_repair_role_probe import BASE, reference_for_batch, verify
from run_landslide_window_optimization import read_json
from write_landslide_dual_reference_report import full_table

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs
from gnss_sim.landslide_review_utility import correction_utility

ROOT = BASE.parents[1]
PROBE = BASE / "repair-role-development-v1"
OUT = BASE / "repair-role-development-report-v1"
TARGETS = ("ai_0133_0144", "ai_0145_0156", "process_0145_0156")


def draw_case(batch, cid, reference, predictions):
    binding = read_json(PROBE / "parent-bindings.json")[batch]
    case = read_json(Path(binding["data"]) / "inputs" / cid / "input.json")
    observed = np.array([row is not None for row in case["displacement_mm"]])
    values = np.array([[np.nan] * 3 if row is None else row for row in case["displacement_mm"]])
    labels = ["Author-AI reference"]
    states = [[row["activity_label"] for row in reference]]
    for rep in (0, 1):
        for label, stem in (("Original", "adaptive_no_feedback"), ("Roles", "repair_roles"),
                            ("Untyped joint", "untyped_joint")):
            method = f"{stem}_r{rep}"
            if stem == "adaptive_no_feedback":
                method += "__conflict_scope"
            labels.append(f"{label} r{rep}")
            states.append(predictions[method][cid]["activity"])
    figure, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 3]})
    for component, name in enumerate(("N", "E", "U")):
        axes[component].plot(values[:, component], linewidth=.7)
        axes[component].set_ylabel(name + " mm")
        axes[component].grid(alpha=.2)
    codes = np.array(states) + 1
    codes[:, ~observed] = 3
    axes[3].imshow(codes, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
        cmap=ListedColormap(["#edc46a", "#e0e7e0", "#407ea0", "#777777"]),
        extent=(-.5, len(values) - .5, len(labels) - .5, -.5))
    axes[3].set_yticks(range(len(labels)), labels, fontsize=9)
    axes[3].set_xlim(0, len(values) - 1)
    axes[3].set_xlabel("Original day; blue=activity, pale=stationary, amber=unknown, gray=missing")
    figure.suptitle(cid + " | exposed development; post-score inspection; frozen reference unchanged")
    figure.tight_layout()
    figure.savefig(OUT / (cid + ".png"), dpi=145)
    plt.close(figure)
    return observed


def inspect_changes():
    rows, changes = [], []
    examples = {"ai_0133_0144": ("case_0139", "case_0143"),
                "ai_0145_0156": ("case_0148", "case_0152", "case_0153", "case_0154")}
    for batch, cids in examples.items():
        labels, observed, _ = reference_for_batch(batch)
        predictions = read_json(PROBE / "analysis" / batch / "predictions.json")
        for cid in cids:
            draw_case(batch, cid, labels[cid], predictions)
        for cid in labels:
            actual = np.array([row["activity_label"] for row in labels[cid]])
            for rep in (0, 1):
                treatment = np.array(predictions[f"repair_roles_r{rep}"][cid]["activity"])
                for stem in ("adaptive_no_feedback", "untyped_joint"):
                    control = f"{stem}_r{rep}"
                    if stem == "adaptive_no_feedback":
                        control += "__conflict_scope"
                    before = np.array(predictions[control][cid]["activity"])
                    utility = correction_utility(actual, before, treatment, observed[cid])
                    rows.append({"batch": batch, "case_id": cid, "repeat": rep, "control": control,
                        **utility,
                        "tp_difference": int(((actual == 1) & (treatment == 1)).sum()
                            - ((actual == 1) & (before == 1)).sum()),
                        "fp_difference": int(((actual == 0) & (treatment == 1)).sum()
                            - ((actual == 0) & (before == 1)).sum())})
                    for truth in (-1, 0, 1):
                        for old in (-1, 0, 1):
                            for new in (-1, 0, 1):
                                if old == new:
                                    continue
                                mask = observed[cid] & (actual == truth) & (before == old) & (treatment == new)
                                for start, stop in runs(mask):
                                    changes.append({"batch": batch, "case_id": cid, "repeat": rep,
                                        "control": control, "reference": truth, "before": old,
                                        "after": new, "start": start, "stop": stop})
    write_csv(OUT / "per-record-corrections.csv", rows)
    write_csv(OUT / "changed-intervals.csv", changes)


def draw_summary(tables):
    arms = ("adaptive_no_feedback", "repair_roles", "untyped_joint",
            "roles_placement_only", "roles_authority_only")
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for axis, (target, rows) in zip(axes, tables.items(), strict=True):
        table = {row["method"]: row for row in rows}
        for rep, offset, color in ((0, -.18, "#407ea0"), (1, .18, "#bd8744")):
            names = [f"{arm}_r{rep}" + ("__conflict_scope" if arm == "adaptive_no_feedback" else "")
                for arm in arms]
            axis.barh(np.arange(len(arms)) + offset, [float(table[name]["f1"]) for name in names],
                height=.34, label=f"r{rep}", color=color)
        axis.set_yticks(range(len(arms)), ["Original", "Repair roles", "Untyped joint",
            "Placement only", "Authority only"])
        axis.set_xlim(.7, 1)
        axis.set_xlabel("Activity F1")
        axis.set_title(target)
        axis.grid(axis="x", alpha=.2)
        axis.legend()
    axes[0].invert_yaxis()
    figure.suptitle("Stopped development: improvements do not preserve weak recall on both batches")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(OUT / ("comparison." + suffix), dpi=155)
    plt.close(figure)


def report():
    verify(PROBE)
    verify_manifest(PROBE / "analysis")
    OUT.mkdir(exist_ok=False)
    (OUT / "source.py").write_bytes(Path(__file__).read_bytes())
    tables = {}
    for name in TARGETS:
        with (PROBE / "analysis" / name / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    inspect_changes()
    draw_summary(tables)
    audit = read_json(PROBE / "analysis/audit.json")
    text = """# 按修复用途配置观察与修订范围：停止的开发试验

2026-10-11。结论：**STOPPED**，不进入未见确认、不替换默认方法。两批完整作者AI开发参考上的8个预登记比较仅5个通过；不能将旧批次改善或净确定纠错为正写成全面成功。

## 机制与协议

保持原21/42日原始三块持续运动冲突、14日准入、每记录每遍0/1/2个不重叠180日窗口、图像、提示和冻结阶段头。漏运动冲突全部保留；已有视觉活动的否定性冲突只在其事件首尾49日附近用于排序。按这些修复线索选择与原方法等次数的最优不重叠窗口组合；线索再扩49日建立修订权限。因此有效边界权限可达98日，并非缩窄到49日。数值仍不直接赋活动。

6个新增方案各2遍：完整repair_roles、只改变位置、只改变权限、等数量居中权限、按原冲突总量联合排序、联合排序加新权限。最后两项用于区分修复语义与贪心改为联合优化的收益。内部过合并平台可能失去复核线索，这是预先披露的风险。

数据是已暴露的0133—0144和0145—0156全部记录，无新抽样、再训练、改标签或门槛搜索。第三目标仅在后12条的过程参考描述评分，数值控制及共同阶段头使用此前相同6训练/6验证、适配该目标的冻结模型。全部34原方法保留，加12新方案，46方法×3目标。调用预算在48个记录/遍组合完全相同；位置仅4组变化，具体在structure-check.json。

实验前冻结继续条件：两AI批次、两遍完整方案相对原方法均F1严格提高、弱召回不降、FP不增、净确定纠错>0；相对原冲突联合排序均F1和弱召回不降、净纠错非负。不通过就停止，无seed/门槛/重复救分。

## 真实结果与解释

旧批次原方法F1 .8964/.9029，新方案.9301/.9212，FP213/245降到145/181，弱召回.7968/.8777升到.8777/.8777，净确定纠错+123/+64。然而无类型联合排序已达.9301/.9295，第二遍弱召回.9299高于新方案.8777；新方案相对它净−24，未证明修复语义优于简单组合选择。

新批次原方法F1 .8536/.8363，新方案.8649/.8359。FP207/187降到158/138，但弱召回.6230/.5234降到.6035/.4531，分别损失10/36个弱活动日。净确定纠错+39/+13也不能掩盖这个代价。只改权限第一遍F1 .8636、弱召回.6699且不减少FP；只改位置两遍F1 .8547/.8359，弱召回.5566/.4531。不同机制不具有一致协同，不能选择较好的消融作为本轮预登记主结果。

过程目标原方法F1 .8280/.8073，新方案.8340/.7976，弱召回.5368/.4732降到.4806/.3533；第二遍净确定纠错−2。与AI目标分母不同，不跨目标比较召回数值。事件F1未改变；阶段分数仍来自共同数值头和活动门控，不称视觉阶段推理收益。

原34方法×2AI目标的68组历史分数重现；3目标仍报告全部活动P/R/F1、事件IoU .3/.5、边界MAE、阶段内部/端到端F1、弱召回和调用/token。完整结果见下表及CSV。

## 成本与复现

66个需要的局部问题中63个图像哈希和提示完全相同的历史回答复用，只新增3启动/3回执、12,187报告token，10.75秒；不重试。两种日期的回答不能构成同时期fresh control，也不支持排除模型时间漂移。缓存开发总请求与部署逐方法请求分开，前者不可冒充部署成本。

`scripts/run_landslide_repair_role_probe.py`按prepare→run→analyze执行，目标目录必须不存在。默认源码和旧冻结产物未改。冻结记录、协议、来源快照、schedule、payload/response、预测和46组评分保存在repair-role-development-v1；本报告图为固定预测后的事后误差检查，不回写作者参考。

## 全部统一对比

"""
    for target, rows in tables.items():
        text += f"\n### {target}\n\n" + full_table(rows) + "\n"
    text += "\n下一步应改变可证伪的识别机制假设：本轮同一局部回答会同时撤平台误报并撤弱活动，单靠已有窗口的重新分配不能解决这类证据混合。不复活已停止方案，也不将联合组合优化单独包装成新的视觉/Agent贡献。\n"
    report_path = ROOT / "docs/landslide-repair-role-development-results.md"
    report_path.write_text(text, encoding="utf-8")
    write_json(OUT / "report-binding.json", {"path": str(report_path), "sha256": sha(report_path),
        "audit": audit, "decision_sha256": sha(PROBE / "analysis/decision.json")})
    write_json(OUT / "manifest.json", {"files": {
        path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file()}})


if __name__ == "__main__":
    report()
