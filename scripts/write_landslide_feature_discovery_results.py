"""Publish the fixed-pool negative result without changing historical evidence."""

from __future__ import annotations

from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha


def write():
    run = BASE / "feature-discovery-calibration-v1"
    analysis = BASE / "feature-discovery-analysis-v3"
    archive = BASE / "manuscript-v49"
    decision = read_json(run / "decision.json")
    if decision["decision"] != "STOP_FIXED_GENERATED_POOLS":
        raise ValueError("This writer documents the stopped fixed pools")
    names = ("landslide-adaptive-review-manuscript-draft", "landslide-adaptive-review-method",
             "landslide-submission-roadmap", "project-status")
    for name in names:
        if (ROOT / "docs" / (name + ".md")).read_bytes() != (archive / (name + "-v48-original.md")).read_bytes():
            raise ValueError("Archive original v48 bytes before updating once")
    if not read_json(archive / "v48-preservation-before.json")["all_match"]:
        raise ValueError("Previous delivery not preserved")
    costs = read_json(analysis / "cost.json")
    if costs["new_reported_tokens"] != 24540:
        raise ValueError("Cost changed")
    tables = (analysis / "results.md").read_text(encoding="utf-8")
    rows = csv_rows(analysis / "summary.csv")
    controls = ("base18", "base_expert", "base_blocks", "visual_numeric-r0_union",
                "visual_numeric-r1_union", "visual_only-r0_union", "visual_only-r1_union",
                "numeric_only-r0_union", "numeric_only-r1_union")
    brief = ["| 方法 | 过程F1 | 弱R | FP/FN | 净纠错日 | 作者F1 | 作者净纠错日 |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for name in controls:
        primary = next(row for row in rows if row["method"] == name and row["target"] == "primary_process")
        secondary = next(row for row in rows if row["method"] == name and row["target"] == "secondary_developer")
        brief.append(f"| {name} | {float(primary['f1']):.6f} | {float(primary['weak_recall']):.6f} | "
                     f"{primary['fp']}/{primary['fn']} | {primary['net_vs_base']} | "
                     f"{float(secondary['f1']):.6f} | {secondary['net_vs_base']} |")
    mechanism = r"""
最新特征池研究[36]限定VLM从少量调参图提出统计函数，再固定检测器检验；一般特征池生成已有先例。本轮将其落实为一个离线最小候选：生成器只见六个不同过程训练记录窗，静稳/弱活动/其他活动各二例及中心标签；视觉＋数值、纯视觉、纯数值三输入臂，各两遍、每池三函数。未提供验证记录、拟合检测器或分数，但作者有历史开发暴露，不能称整体盲设计。

令 $x_i\in\mathbb R^3$ 为原NEU位移，查询原日 $t$ 的十三个14日块为

$$
B_j(t)=\{t-91+14j,\ldots,t-78+14j\},\quad j=0,\ldots,12.
$$

每块至少八个完整原NEU观测才计算逐轴中位数，否则该槽保留NaN；窗外和缺测不插值、不压缩。以第一个可用块去参考偏置，再除原日差噪声代理，得到

$$
z_j(t)=[m_j(t)-m_{j_0}(t)]\oslash\sigma,\qquad
j_0=\min\{j:m_j(t)\text{可用}\}.
$$

全部块不可用时全部特征未知。中心在槽6，块中心为原日偏移−84.5＋14j。这是离线双侧观察，不是实时因果预警；噪声单位不是置信概率。

先逐一定义审核再执行：18函数中四个超过冻结12行上限，包含一个跳过缺测增量后将压缩第5槽误认中心、且可能越界的函数。统一剔除，原答复不修复、不重生成。保留14函数，六池维度1/3/1/3/3/3（按数值r0/r1、主r0/r1、视觉r0/r1）；视觉r1有一个语义重复，原样保留并披露。主r0只剩中心增量范数，与数值r0相同；主r1为中心增量范数、前后方向余弦、二阶差分范数，其“加速度”未按物理时间缩放，不直接赋A/S/D。

比较原18维、固定六通用统计单独/合并、全部39块值＋13可用标志单独/合并，以及六生成池单独/合并，共17臂。通用统计为前后增量范数、前后余弦、两块端点净变化、二阶差分范数和窗内连续增量范数中位数；端点净变化只是描述统计，不补造中间缺测。全部块控制与生成函数共用观察范围。所有臂同原24/12和6/6划分、全部原已知活动支撑训练日、120树、深度2/4、学习率.05、seed20261007及19阈值，每臂38机会，以日F1＋事件F1首个最大选择。保留原14日持续性、缺测、活动/阶段支撑及对应训练阶段头，不逐特征筛选、不调seed或窗口。
"""
    outcome = r"""
两遍主合并F1 .917398/.920437低于原强基线.924425，净纠错−22/−12日，弱R .796186/.808105亦低于.843862。主r0减少50个FP但增加72个FN；r1减少33个FP、增加45个FN。作者六验证两遍均.947668低于.949461、净−3日，弱参考分母零保持缺项。登记条件5/14通过，STOP_FIXED_GENERATED_POOLS；不扩展固定候选、不放宽审核再救分。

视觉必要性四条件仅2/4通过：主r0与纯数值r0概率/预测相同，主r1也未优于视觉r1。通用数值合并F1 .936436、弱R .892729、净＋50日（五记录），作者.950057、净＋1；其FP增加13日，事件F1 .3仍.750000。这是新增数值描述的开发改善，不能归因于生成器或声明VLM方法贡献。全部块合并过程F1 .870481、作者.957746，说明同输入的表示/泛化差异；不能只挑过程较弱控制称胜出。

实际四图和原14日NEU中位数复核：0031单次偏移后平台被主r1新增24日活动；0036修正部分平台/端点但漏掉弱累计；0059第二段提前终止损3个确定活动日，另外14日参考未知不计对错；0057输出不变。诊断按事后净纠错极值选取，不改原参考或登记人口，不是独立专家确认。完整P/R/F1、事件IoU .3/.5、边界MAE、内部/端到端阶段、逐记录及家族见完整报告。

六新启动/六回执，24540报告token，API分段42.937秒；68新模型、1292配置全部保存，训练原已知支撑15319/3118日。拟合及验证概率计算合计6.526秒，共享预处理22.096秒包含原接口额外栅格，未将其称纯数值零图成本；评分、I/O和完整端到端CPU不在这些分段内。168形状探针、六payload重建、34选择重放、223380日历缺测/阶段/支撑、sklearn日计数及事件边界一致性通过。20相关测试和新增源码Ruff通过。

生成准备v1仅移除unused import，0API，v1/v2输入图/表/提示/调度字节相同；报告v1因NumPy冷初始化、v2因JSON池键排序改变探针列表顺序失败，均在评分前、0API。v3在受限代码外初始化NumPy并按池/特征/形状语义键比对；原函数、训练权重和所有冻结预测未改。
"""
    result = ("# 离线视觉特征发现：固定池强基线比较与停止决定\n\n"
              + mechanism.strip() + "\n\n" + "\n".join(brief) + "\n\n" + outcome.strip()
              + "\n\n" + tables[tables.index("## primary_process"):]
              + "\n## 复现与后续研究\n\n"
              "PYTHONPATH=src;scripts、PYTHONIOENCODING=utf-8，使用既有environment/Scripts/python.exe。"
              "probe_landslide_feature_discovery.py prepare/calibrate --out新目录；"
              "report_landslide_feature_discovery.py --run冻结校准目录 --out新目录可无API权重重放。"
              "原生成六池保留在visual-feature-proposals-v2；审核/模型/概率/预测在feature-discovery-calibration-v1；"
              "主报告为feature-discovery-analysis-v3，实际复核为feature-discovery-visual-review-v1。\n\n"
              "通用数值控制是更强比较候选，但本轮未新做规则/PELT/直接VLM/固定复核的完整独立确认，"
              "不能替代初稿原统一完整基线，也不能升级为核心贡献。"
              "下一轮优先研究局部方向证据如何保留弱累计而排除单次偏移，"
              "若引入新的Agent/VLM决策机制必须另冻假设，包含此更强数值控制和新独立数据；"
              "不继续同池提示搜索。累计3895启动/3893回执，至少16853587报告token，"
              "受控192/半合成60不变。goal active，稳定方法贡献和独立可信确认未完成。\n")
    (ROOT / "docs/landslide-feature-discovery-results.md").write_text(result, encoding="utf-8")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    manuscript = manuscript.replace("完整中文初稿v48", "完整中文初稿v49", 1)
    manuscript = manuscript.replace("更新前v47主稿及三状态文档原字节存档。新增活动内运动坐标残差阶段小试、四臂两遍与强阶段头比较、八图视觉数值复核及相关文献；候选停止。",
                                    "更新前v48主稿及三状态文档原字节存档。新增有限训练图特征发现、三输入臂两遍固定池、强数值同预算比较及四图视觉数值复核；固定候选停止。", 1)
    manuscript = manuscript.replace("它提供下一轮研究依据，尚未实现新特征方法或证明GNSS贡献。",
                                    "它提供下一节有限离线特征发现的依据；通用框架不能直接作为GNSS首创。", 1)
    addition = ("\n### 6.4.13 有限训练图特征发现与同预算强基线\n\n" + mechanism.strip()
                + "\n\n" + "\n".join(brief) + "\n\n" + outcome.strip()
                + "\n\n详见[完整结果及复现](landslide-feature-discovery-results.md)。"
                "此轮仅开发验证，所有旧停止决定、原完整规则/PELT/VLM/固定Agent对照及参考保持。\n\n")
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", addition + "### 6.5 阶段与计算成本", 1)
    manuscript = manuscript.replace("累计3825启动/3823回执、至少16694503报告token",
                                    "截至v49累计3895启动/3893回执、至少16853587报告token", 1)
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = """
## 2026-10-11：有限视觉特征池停止，初稿v49

[完整结果](landslide-feature-discovery-results.md)。落实最新特征池文献为训练六窗、三输入臂两遍、六真实生成请求。18定义按冻结契约审核保留14，四超12行含一缺测压缩错位；原答复不修复，无再生成/特征筛选。17臂、68新XGB、1292配置，原24/12及6/6、全部已知支撑训练日，每臂38机会；数值/视觉/同块信息控制齐备，阶段头和支撑保持。

主合并过程F1 .917398/.920437低于base18 .924425、净−22/−12日、弱R下降；作者两遍.947668低于.949461、净−3日。5/14条件，STOP_FIXED_GENERATED_POOLS；视觉必要性2/4，r0与纯数值相同。通用数值合并.936436、净＋50、弱R .892729、FP增加13，作者.950057/净＋1，作为更强数值控制证据，不能据此称VLM贡献或独立结论。

四图及原块中位数实际复核：0031偏移后平台新增24日误报，0036部分修正伴弱运动漏检，0059损3已知日/另外14未知不计对错，0057无修订。六payload重建、168探针、34选择/概率/预测重放、223380日历契约及sklearn/事件边界通过；20测试/Ruff通过。原准备v1和报告v1/v2评分前0API失败保留，最终报告v3成功，不改冻结训练或生成定义。

新六启动/六回执/24540报告token，API42.937秒、68模型拟合与验证概率6.526秒、共享预处理22.096秒（含原接口栅格）；非完整端到端。累计3895启动/3893回执、至少16853587token，受控192/半合成60不变。v48四文档原字节存档、448旧文件映射保留，v49公式LaTeX。下一机制另冻假设、以通用数值合并为更强控制、不继续同池提示搜索；goal active，稳定贡献与独立可信确认未完成。

"""
    for name in names[1:]:
        path = ROOT / "docs" / (name + ".md")
        text = path.read_text(encoding="utf-8")
        head, tail = text.split("\n", 1)
        path.write_text(head + "\n" + status + tail, encoding="utf-8")
    export(archive / "progress.json", {"status": "goal_active", "new_models": 68,
                                      "new_generation_calls": 6, "new_trials": 1292,
                                      "decision": decision["decision"],
                                      "independent_confirmation": False,
                                      "actual_visual_reviews": 4,
                                      "formula_format": "MarkdownLaTeX"})
    # Verify all old delivered files; only four current documents map to archives.
    old = read_json(archive / "v48-delivery-manifest-original.json")["files"]
    mapping = {f"docs/{name}.md": archive / (name + "-v48-original.md") for name in names}
    checks = {name: sha(mapping.get(name, ROOT / name)) == expected for name, expected in old.items()}
    if not all(checks.values()):
        raise ValueError("Historical delivery changed outside archived documents")
    export(archive / "v48-preservation-after.json", {"all_match": True, "count": len(checks),
                                                    "archived_document_mapping": {key: str(value) for key, value in mapping.items()},
                                                    "checks": checks})
    print("Wrote v49 and preserved448 old delivery files with archive mapping")


if __name__ == "__main__":
    write()
