"""Report the stopped repair hypotheses without changing frozen experiment outputs."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from finalize_landslide_adaptive_evidence import verify_request_receipts
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
ANALYSIS = BASE / "missed-motion-repair-analysis-v2"


def number(value):
    return f"{float(value):.4f}" if value not in (None, "") else "—"


def metrics_table(rows):
    lines = ["| 配置 | P | R | F1 | 弱R | FP | 事件F1 .3/.5 | 事件数 | 起/止MAE日 | 活动内/端到端阶段宏F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in rows:
        fields = [label, *[number(row[key]) for key in ("precision", "recall", "f1", "weak_recall")],
                  row["fp"], number(row["event_f1_iou03"]) + "/" + number(row["event_f1_iou05"]),
                  row["predicted_events"], number(row["onset_mae_days"]) + "/" + number(row["offset_mae_days"]),
                  number(row["internal_stage_macro_f1"]) + "/" + number(row["stage_macro_f1"])]
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def close():
    out = BASE / "manuscript-v13"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    source = BASE / "background-review-v1"
    verify_freeze(source)
    receipts = verify_request_receipts(source)
    with (ANALYSIS / "all-metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["run"], row["method"]): row for row in rows}
    names = ["missed-motion-repair-development-v2", "calibrated-motion-repair-development-v1",
             "persistent-motion-repair-development-v2"]
    titles = ["逐日双证据", "校准双证据", "校准＋14日连续新增"]
    selected = []
    for rep in (0, 1):
        selected.append((f"直接视觉/{rep + 1}", indexed[names[0], f"direct_visual_r{rep}"]))
        selected.append((f"原整目标复核/{rep + 1}", indexed[names[0], f"adaptive_no_feedback_r{rep}"]))
        for name, title in zip(names, titles):
            selected.append((f"{title}/{rep + 1}", indexed[name, f"adaptive_no_feedback__raw_visual_additions_r{rep}"]))
    for method in ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage", "raw_multiscale_rule",
                   "raw_rule_84", "xgboost_raw_persistence", "xgboost_42_matched_support", "xgboost_84"):
        selected.append((method, indexed[names[0], method]))
    table = metrics_table(selected)
    ablations = []
    for rep in (0, 1):
        for title, variant in (("主连续双证据", "raw_visual_additions"),
                               ("去数值证据", "visual_additions_without_raw"),
                               ("去局部视觉", "numerical_additions_without_visual"),
                               ("去连续约束", "raw_visual_additions_without_continuity")):
            ablations.append((f"{title}/{rep + 1}", indexed[names[2], f"adaptive_no_feedback__{variant}_r{rep}"]))
        ablations.append((f"固定连续双证据/{rep + 1}",
                          indexed[names[2], f"fixed_image_review__raw_visual_additions_r{rep}"]))
    ablation_table = metrics_table(ablations)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#28475e", "#ae5966", "#bc842d", "#24887b"]
    for axis, metric, title in zip(axes, ("f1", "weak_recall", "event_f1_iou03"),
                                    ("Daily F1", "Weak recall", "Event F1 (IoU .3)")):
        for position, (label, name) in enumerate(zip(["Direct", "Exact", "Calibrated", "14-day"],
                                                     [names[0], *names])):
            stem = "direct_visual" if position == 0 else "adaptive_no_feedback__raw_visual_additions"
            values = [float(indexed[name, f"{stem}_r{rep}"][metric]) for rep in (0, 1)]
            axis.scatter([position - .1, position + .1], values, color=colors[position], s=45)
            axis.plot([position - .1, position + .1], values, color=colors[position], linewidth=1)
        axis.set_xticks(range(4), ["Direct", "Exact", "Calibrated", "14-day"])
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Cached exposed development | two repeats shown separately | no new model calls")
    fig.tight_layout()
    fig.savefig(out / "repair-comparison.png", dpi=160)
    plt.close(fig)
    section = f"""### 6.17 单向漏运动修订：逐日、校准与连续性开发检验

为检验原背景实验中的收益能否通过更谨慎的修订保留，复用全部24条已见记录、两遍冻结初判、固定／自适应局部回答及九个共同重拟合数值控制。保留每个初判活动日；只在既定选中目标内，原始21／42日运动证据和局部视觉在同一日期均为阳性时新增活动。数值阳性但局部否定／未知时保留未知，不允许数值证据独自建立活动，不扩49日权限。A/S/D仍来自同一冻结训练阶段头并重新门控，不能称原生视觉阶段提升。

首版沿用选窗0.03mm/日、2倍日差MAD噪声比门槛；后续独立开发假设复用既有train24／validation12选出的原始规则0.12mm/日、3倍噪声比，仅改变新增许可，不改原选窗。发现逐日交集造成事件碎片后，再复用数值基线已有14连续观测支撑日约定，限制新增片段，不桥接缺测、不填负日、不扩大边界，也不扫描持续时长。这些都是看过原测试后的开发检验，预测先保存再评分不使它们重新独立。

{table}

逐日双证据两遍新增359／568TP、1962／2318FP，无原有TP损失，F1 .6097／.5806低于直接.6722／.6330。校准版新增136／155TP、361／334FP，F1 .6668／.6331仍未稳定超过直接；第二遍增量仅.000042。逐日版事件从29／28个增至175／206个，校准后仍为112／117个，事件F1降至.1910／.1826或.2647／.2553。匹配边界MAE只对匹配事件计算，不能用较小条件误差掩盖新增碎片和漏配。

14日连续版将事件数降至35／34，但F1 .6737／.6283仅第一遍微升，第二遍下降；只新增71／17TP和115／114FP，第二遍正确新增仅一条记录，事件F1 .6102／.6207仍低于直接.6792／.6923。连续版去局部视觉为.6739／.6275，去数值为.5765／.5524，未建立局部视觉相对数值新增的稳定F1贡献。固定观察、双向精确日期对照以及全部消融均保存，没有改用最好配置作为主方法。三版均未通过各自执行前登记的继续条件，停止这条修订扩展，不在同批调整门槛、连续天数或挑记录救分。

{ablation_table}

本轮新增API请求0；复用的原背景请求有315份回执，并不代表研究从零成本获得这些观察。三批41／41／47配置共129个评分行，含重复基线，不能当129种独立方法；17346已知日、4760活动日、1706弱日不变。216数值预测重算与旧缓存一致，独立日指标／阶段／事件匹配及修订权限核验通过；17520行完整主预测时间轴和24张全记录图保存。六项新增业务测试及九项相关既有测试通过。实测背景与注入真值的限制不变，三版负结果不替代正式领域观测验证。

![单向修订三版与直接视觉的两遍比较](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v13/repair-comparison.png)

"""
    report = f"""# 单向漏运动修订：三项停止的开发假设

2026-10-09。ARS研究执行与验证；已完成缓存开发实验，不是独立确认。三项假设均未通过各自继续条件，不接入主方法，不新增VLM调用。独立领域复核者暂缺，按用户授权继续自主研究，不生成专家标签。

{section}

## 证据与复现

完整评分、逐记录／家族／四背景组、事件IoU.3／.5、边界、真实活动内与端到端阶段见[分析目录](../artifacts/landslide-adaptive-review-2026-10-09/missed-motion-repair-analysis-v2/all-metrics.csv)。三批冻结预测、许可、输入哈希、继续判定分别在missed-motion-repair-development-v2、calibrated-motion-repair-development-v1、persistent-motion-repair-development-v2。原始全局图／真实请求仍在background-review-v1，无新图送模、无新提示、无重训。

首版missed-motion-repair-development-v1标签路径错误，评分前失败，预测保留；v2修正为原analysis-v1/reference-subset.csv，并用真实固定窗索引恢复目标。连续版v1仅import排序Lint失败，缓存执行已完成；v2只修新源排序，两版predictions／permissions／results／decision逐字节相同。旧失败源、旧评分与旧冻结文件全部保留，没有回写旧运行。最终129配置的分析为analysis-v2，analysis-v1是前两批有效历史结果。

最新文献阅读范围仍见[最近邻核查](landslide-nearest-method-code-review.md)及stage-consistency-search-v1。MORA等已有受保护修订先例，单向保留、通用共识或14日持续规则本身不主张首创；本轮也未提供足够识别收益支持新贡献。

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD/scripts"
$experimentPython = 'artifacts/reproduction-2026-10-07/environment/Scripts/python.exe'
$artifactRoot = 'artifacts/landslide-adaptive-review-2026-10-09'
# 输出目录必须不存在；以下只读重算，不发送模型请求。
& $experimentPython scripts/summarize_landslide_missed_motion_repair_v2.py --runs "$artifactRoot/missed-motion-repair-development-v2" "$artifactRoot/calibrated-motion-repair-development-v1" "$artifactRoot/persistent-motion-repair-development-v2" --out "$artifactRoot/missed-motion-repair-analysis-replay"
& $experimentPython -m pytest tests/test_landslide_missed_motion_repair.py tests/test_landslide_persistent_motion_repair.py tests/test_landslide_revision_scope.py tests/test_landslide_adaptive_review.py -q
```

后续研究需先验证“连续运动提案相对于背景能否区分”的观察价值，不能继续把滚动证据日期直接当活动边界，也不能靠保留全部旧阳性宣称可靠性提高。原强后弱场景的局部观察价值仍可作为研究依据，新的算法选择必须另作冻结验证。
"""
    draft_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous = draft_path.read_bytes()
    if previous != (BASE / "manuscript-v12/manuscript-v12.md").read_bytes():
        raise ValueError("Current manuscript differs from frozen v12")
    (out / "manuscript-v12-original.md").write_bytes(previous)
    draft = previous.decode("utf-8")
    draft = draft.replace(
        "> 完整中文初稿v12，2026-10-09。新增活动内原始速率矛盾的阶段开发及12条新冻结确认：开发修好105/46日、无正确损失，新批选择器两遍0触发，不升级该机制。各24配置、全部失败和强基线保留；独立领域观测参考仍缺，本稿不是投稿定稿。",
        "> 完整中文初稿v13，2026-10-09。新增单向漏运动修订的逐日、校准和连续性三项开发实验：129评分行均保留，三版未有稳定收益，停止扩展；新增API请求0。此前阶段确认0触发、全部失败和强基线继续保留。独立领域观测参考仍缺，本稿不是投稿定稿。",
    )
    draft = draft.replace("## 7 讨论", section + "## 7 讨论", 1)
    chinese = "追加单向修订研究保留全部旧阳性并要求同日原始运动与局部视觉一致，仍未稳定改善活动识别；复用训练校准门槛和14日连续约定后也未通过继续条件。逐日交集显著增加事件碎片，不能用召回增加替代完整活动、事件与成本评价。"
    english = "A subsequent one-sided repair preserves initial positives and requires same-date raw/visual agreement. Neither calibrated evidence nor the existing fourteen-day continuity convention yields a stable improvement; exact-date intersections substantially fragment events. These extensions are stopped and not presented as validated contributions."
    draft = draft.replace("**关键词：**", chinese + "\n\n**关键词：**", 1)
    draft = draft.replace("## 1 引言", english + "\n\n## 1 引言", 1)
    draft = draft.replace("## 8 可复现性、完整性与AI使用说明",
                          chinese + "\n\n## 8 可复现性、完整性与AI使用说明", 1)
    draft = draft.replace("| 当前全文与v11原字节快照、补充事件边界和成本 | manuscript-v12 |",
                          "| 历史全文v12与v11原字节快照、补充事件边界和成本 | manuscript-v12 |\n"
                          "| 单向修订三项开发假设、129配置及全部停止判定 | missed-motion-repair-analysis-v2 |\n"
                          "| 当前全文与v12原字节快照、单向修订图 | manuscript-v13 |")
    draft_path.write_text(draft, encoding="utf-8")
    (out / "manuscript-v13.md").write_text(draft, encoding="utf-8")
    (ROOT / "docs/landslide-missed-motion-repair-results.md").write_text(report, encoding="utf-8")
    prior_ledger = read_json(BASE / "manuscript-v12/run-ledger-end.json")
    save_json(out / "run-ledger-end.json", {
        **prior_ledger, "new_requests_started": 0, "new_completed_local_receipts": 0,
        "new_valid_model_json": 0, "new_reported_tokens": 0,
        "provider_rejections": 0, "connection_failures": 0,
        "cached_background_receipts_verified": len(receipts),
        "new_records": 0, "goal_status": "active", "round_status": "progress",
    })
    save_json(out / "checks.json", {
        "analysis": read_json(ANALYSIS / "state.json"),
        "v12_snapshot_exact": (out / "manuscript-v12-original.md").read_bytes() == previous,
        "cached_background_receipts": len(receipts),
        "tests": "6 new and9 related tests passed in recorded separate runs; no failures",
        "lint": "final source versions passed; persistent v1 import-order failure retained",
        "reference": "known injection, previously exposed; no expert labels",
        "files": {str(path.relative_to(out)): sha(path) for path in out.iterdir() if path.is_file()},
    })
    print({"manuscript": str(out), "cached_receipts": len(receipts), "new_api_requests": 0}, flush=True)


if __name__ == "__main__":
    close()
