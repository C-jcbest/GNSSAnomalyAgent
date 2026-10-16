"""Write the direction development findings without promoting a failed mechanism."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def number(value):
    return f"{float(value):.4f}" if value not in (None, "") else "—"


def comparison_table(rows):
    lines = ["| 配置 | P | R | F1 | 弱R | FP | 覆盖 | 事件F1 .3/.5 | 起/止MAE日 | 活动内/端到端ASD宏F1 | 回执请求/token |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["method"], *[number(row[key]) for key in ("precision", "recall", "f1", "weak_recall")],
                  row["fp"], number(row["coverage"]),
                  number(row["event_f1_iou03"]) + "/" + number(row["event_f1_iou05"]),
                  number(row["onset_mae_days"]) + "/" + number(row["offset_mae_days"]),
                  number(row["internal_stage_macro_f1"]) + "/" + number(row["stage_macro_f1"]),
                  row["requests"] + "/" + row["reported_tokens"]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def close():
    out = BASE / "manuscript-v15"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    versions = ["direction-review-development-v1", "causal-direction-review-development-v1"]
    rows = {}
    for version in versions:
        with (BASE / version / "analysis-v1/summary.csv").open(encoding="utf-8", newline="") as handle:
            rows[version] = list(csv.DictReader(handle))
        if read_json(BASE / version / "decision.json")["allows_new_confirmation"]:
            raise ValueError("This closure is specifically for two stopped direction hypotheses")
    indexed = {version: {row["method"]: row for row in values} for version, values in rows.items()}
    causal = indexed[versions[1]]
    primary = [causal[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
               ("direct_visual", "old_adaptive_image", "old_fixed_image", "direction_main", "direction_fixed")]
    primary += [row for row in rows[versions[1]] if not row["method"].endswith(("_r0", "_r1"))]
    ablations = [causal[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                 ("direction_main", "direction_without_history", "direction_without_motion",
                  "direction_visual_only", "direction_numeric_text")]
    old = indexed[versions[0]]
    old_comparison = [old[f"{stem}_r{rep}"] for rep in (0, 1) for stem in ("direction_main", "direction_fixed")]
    old_comparison += [old[name] for name in ("direction_rule", "xgboost_direction_information")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    stems = ["direct_visual", "direction_main", "direction_without_history", "direction_fixed"]
    titles = ["Direct", "Direction", "No history", "Fixed direction"]
    for axis, metric, title in zip(axes, ("f1", "weak_recall", "coverage"),
                                    ("Activity F1", "Weak recall", "Decision coverage")):
        for position, stem in enumerate(stems):
            axis.plot([position - .1, position + .1], [float(causal[f"{stem}_r{rep}"][metric]) for rep in (0, 1)],
                      marker="o", linewidth=1)
        axis.set_xticks(range(4), titles, rotation=15)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Exposed cached development | no new VLM calls | both direction hypotheses stopped")
    fig.tight_layout()
    fig.savefig(out / "direction-review-comparison.png", dpi=150)
    plt.close(fig)
    section = f"""### 6.19 历史强运动方向与整段视觉提案：没有确认修正价值

历史压缩不能区分弱尾与结束后背景，因此另提出一个几何假设：若局部视觉认为仍有弱运动，该段两次原始位移增量应沿此前强运动方向持续累积。它不是新Agent架构，不修改既有图、提示或回答。强锚点来自共同训练／验证已校准的raw42规则（.12mm/日、噪声代理比3），至少14连续观测支撑日。对原49日权限内、初判尚未阳性的局部视觉阳性提案，取首／中／尾三个不相交14日原始NEU中位数块；提案至少42日，每块至少8完整观测，两次增量都须与锚点余弦≥.5、投影速率≥.03mm/日且投影变化／既有轴噪声合成代理≥2。六十度是固定工程几何容差，后两门槛复用原选窗约定，不是误差概率界限或经本批调优值。提案整段接纳或保留未知，全部初判阳性保留；只按既有≤7日未知桥接聚合缺测提案，缺测本身不赋活动，已知阴性空档不连接。阶段仍为共同训练的数值阶段头，不称原生视觉A/S/D。

首版要求此前完整强支撑段结束后建立锚点，两遍都没有接纳任何主组提案。第一遍36提案中14无前置锚点、11短于证据分辨率、11原始证据冲突；第二遍39提案为14／12／13。另存前缀版本，允许强运动仍持续时，只以此前已完成的最近14个连续强支撑中心估计锚点；所有42+7日原始覆盖先于提案起点，不等待整段结束。它只改变锚点可用时刻，不改变几何／运动门槛，并另行重拟合同信息控制。前缀版第一遍13／11／12，第二遍14／12／13，仍没有接纳主组提案。这两版都是已见开发假设，数据及参考没有重新获得独立性，也不在失败后继续搜索本批门槛。

{comparison_table(primary)}

强支撑门槛仍使用整记录日差MAD噪声代理，原数值矩阵与初判也使用完整记录；这里的前缀仅约束方向增量的原始观测覆盖，不构成整个系统的在线无泄漏推断。前缀测试固定噪声代理，只验证未来原始位移不会改写此前已可用的方向取值。

两版主组两遍F1 .6722／.6330、弱召回.6178／.3910、FP2648／2269与直接视觉相同，新增／损失TP、新增／撤回FP均为0；但**阳性集合相同不等于输出相同**。阴性改为未知使主组覆盖从直接.9529／.9685降至.6359／.5933。第一遍事件F1由.6792变.6923来自未知空档改变七日事件桥接，不能当作新增活动识别收益，结束边界MAE反从36.67增至38.17日；第二遍均未改进。端到端阶段宏F1同直接.6354／.6106，阶段覆盖和未知仍需公开。两遍登记继续条件均失败；数值反馈组也未取得稳定增量。不将保守拒绝、无TP损失或事件数改变包装为可靠性贡献。

{comparison_table(ablations)}

去历史方向版新增117／449TP，同时新增666／1252FP，F1 .6477／.6181，低于直接。去运动幅值条件第一遍增54TP和32FP、第二遍0增益；纯视觉整提案版新增404／637TP和2945／3319FP，F1 .5727／.5466。固定方向版前缀两遍各增86TP、3FP，F1 .6821／.6442，收益均集中于case2060，覆盖.7247／.6845且结束MAE42.63／52.53日，比直接36.67／42.67日较差。首版固定组第二遍还会接纳case2029，增22TP和139FP，前缀锚点拒绝该错误提案。因此方向工具在特定记录可能有纠错价值，尚未证明自适应观察的位置优势；不在看到固定组更高后替换登记主组。

{comparison_table(old_comparison)}

公平控制包括九个原共同train24／validation12数值规则、PELT和XGBoost，以及两版各自的方向规则／增广XGBoost。新规则保留校准强运动，并加入前置方向一致的弱支撑；按3速率×3噪声比九试次选择。增广XGBoost得到原21／42／84矩阵、raw42三轴增量、此前锚点方向、两次余弦／投影速率／噪声代理比及锚点年龄，按原2深度×3概率门槛六试次训练选择。没有视觉回答或开发参考作为训练特征，学习器仍可利用原始数值信息；特征设计发生在已见开发中，不能称独立验证。完整段规则选择.06／2，增广XGB选择深度2／概率.65，F1 .7078／.6802。前缀规则选择.03／2，XGB深度2／概率.5，F1 .7112／.7350。前缀XGB比原XGB84 .7261提高.00884，弱召回.1518对.1290，FP276对294；事件F1却由.5965降至.5763，起／止MAE由13.41／24.24增至16.41／26.76日。它是有日识别收益且事件代价较大的数值开发控制，不能替代Agent/VLM核心贡献或当新独立成绩。

case2028结束后背景提案与强运动方向冲突，可拒绝平台扩张；case2030第一遍270—450日提案在前缀锚点下两次余弦为−.062／−.473，说明背景叠加后真实弱运动并不保证原始局部变化沿历史方向。631—720日提案虽然方向一致，第一次投影／噪声比仅1.44，仍拒绝。历史方向无法解决观测背景与注入运动混合的不可分辨性；更不能假定真实复杂滑体不转向或残差背景必然非活动。该控制限制适用于本批已知注入恢复，不是现场总运动标签。

每版29评分配置、288权限／提案重放、264数值预测重算、315旧payload回执、290家族／背景分组、508080逐日行及24全记录图完成，重复基线不当58个独立方法。6项新业务测试通过，包括改变150日以后的观测不改变150日以前可用锚点；最终源码Ruff通过。两版缓存探针墙钟11.48／10.47秒包含特征计算、绑定核验及评分，不等于部署纯推理延迟；训练CPU未隔离计时，不报零总计算成本。主组完整已有回执72／72请求、254639／254820token；直接24／24、59470／59436。此次新VLM请求0、记录0，累计仍1809启动／1808回执／至少7977130已报告token；108纯受控与60观测残差半合成分开。保留全部冻结源、模型、失败版本及前稿。

本轮另以官方arXiv检索8条，Crossref方向书目查询因证书校验失败，失败原样记录。对《LLM-Generated Feature Pools for Time Series Anomaly Detection》（2609.21801v1）实际读摘要及3.1—3.5：它在调参窗口图上生成特征池，并严格在调参划分选择，最终检测为数值MAD；未定位作者代码，未复现。将LLM生成特征再交数值模型也已有先例，不能因本轮方向XGB提高而改称新Agent。通用缩放／有限循环／特征生成均不作为首创。现无独立领域复核者，用户已授权继续自主研究；本轮两版停止，不升级、不扫阈值或选最佳消融。后续仍需可证伪的视觉纠错机制及新冻结验证，而不是只增加负结果报告。

![方向约束的识别与覆盖取舍](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v15/direction-review-comparison.png)

"""
    report = "# 历史方向约束开发：主组无识别增量，覆盖下降\n\n" + section
    report += "\n## 复现与产物\n\n"
    report += "两版完整协议、预测、锚点、提案、逐日结果、分组及24全图分别位于direction-review-development-v1/analysis-v1和causal-direction-review-development-v1/analysis-v1。前者使用direction-information-controls-v1，后者causal-direction-information-controls-v1。旧冻结产物不要覆盖；以下输出目录必须不存在。\n\n"
    report += """```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$experimentPython='artifacts/reproduction-2026-10-07/environment/Scripts/python.exe'
& $experimentPython scripts/analyze_landslide_direction_review.py --run artifacts/landslide-adaptive-review-2026-10-09/causal-direction-review-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/causal-direction-review-replay --causal
& $experimentPython -m pytest tests/test_landslide_direction_review.py -q
```
"""
    draft_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous = draft_path.read_bytes()
    if previous != (BASE / "manuscript-v14/manuscript-v14.md").read_bytes():
        raise ValueError("Current draft differs from preserved corrected v14")
    (out / "manuscript-v14-original.md").write_bytes(previous)
    draft = previous.decode("utf-8")
    draft = re.sub(r"> 完整中文初稿v14，[^\n]+", "> 完整中文初稿v15，2026-10-09。新增历史方向整提案约束与前缀锚点两版开发、同信息数值控制和完整消融；主组无识别增量且覆盖下降，均停止。新VLM请求0；独立领域观测参考仍缺，本稿不是投稿定稿。", draft, count=1)
    draft = draft.replace("## 7 讨论", section + "## 7 讨论", 1)
    addition = "历史方向与前缀锚点开发两版主组均无识别增量，覆盖下降；固定组收益集中于一条记录。同信息方向XGBoost有小幅日F1收益及事件／边界代价，不将数值特征改进替代Agent贡献。"
    draft = draft.replace("**关键词：**", addition + "\n\n**关键词：**", 1)
    draft = draft.replace("## 1 引言", "Two development tests of historical-direction validation add no positive detections and reduce decision coverage. Fixed-window gains are confined to one record. A direction-augmented XGBoost control improves daily F1 slightly while worsening event and boundary metrics; this numerical result does not establish an Agent contribution.\n\n## 1 引言", 1)
    draft = draft.replace("| 当前全文与v13原字节快照、尺度准入图 | manuscript-v14 |",
                          "| 历史全文v14与v13原字节快照、尺度准入图 | manuscript-v14 |\n"
                          "| 两版方向约束29配置、同信息规则／XGB与停止条件 | direction-review-development-v1；causal-direction-review-development-v1 |\n"
                          "| 当前全文v15、精确v14原字节与方向比较图 | manuscript-v15 |")
    draft_path.write_text(draft, encoding="utf-8")
    (out / "manuscript-v15.md").write_text(draft, encoding="utf-8")
    (ROOT / "docs/landslide-direction-review-results.md").write_text(report, encoding="utf-8")
    (out / "direction-report.md").write_text(report, encoding="utf-8")
    ledger = read_json(BASE / "manuscript-v14/run-ledger-end.json")
    save_json(out / "run-ledger-end.json", {**ledger, "new_requests_started": 0,
                                            "new_completed_local_receipts": 0, "new_reported_tokens": 0,
                                            "new_records": 0, "round_status": "progress", "goal_status": "active"})
    save_json(out / "checks.json", {"analyses": {name: read_json(BASE / name / "analysis-v1/state.json") for name in versions},
                                    "decisions": {name: read_json(BASE / name / "decision.json") for name in versions},
                                    "v14_snapshot_exact": (out / "manuscript-v14-original.md").read_bytes() == previous,
                                    "new_tests_passed": 6,
                                    "files": {path.name: sha(path) for path in out.iterdir() if path.is_file()}})
    print({"manuscript": str(out), "configs_per_version": [len(rows[name]) for name in versions],
           "new_VLM_requests": 0}, flush=True)


if __name__ == "__main__":
    close()
