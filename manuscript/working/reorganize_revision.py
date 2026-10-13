"""Apply the author's six-section structure to the reviewed chapter sources.

This one-off revision moves existing evidence without rerunning experiments.
The original seven-section sources remain recoverable from commit f37f140.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
WORKING = ROOT / "working"


def main():
    results_path = CHAPTERS / "05-results.md"
    results = results_path.read_text(encoding="utf-8")
    if not results.startswith("# 5 实验结果与分析"):
        raise ValueError("Expected the reviewed seven-chapter source before revision.")
    pieces = re.split(r"(?m)^## 5\.\d+ .+\n", results)
    overall, types, seeds, ablation, boundaries, uncertainty, cases = pieces[1:]

    transient_marker = "[[FIGURE:case_transient]]"
    deformation_marker = "[[FIGURE:case_deformation]]"
    case_intro, rest = cases.split(transient_marker, 1)
    transient, rest = rest.split(deformation_marker, 1)
    deformation, failures = rest.split("同一固定选样规则下", 1)
    case_intro = case_intro.strip()

    overall += "\n从对照关系看，基础分类器比较回答数值特征本身的识别能力，候选视觉比较回答补充形态信息后的效果，同响应变体则回答最终规则应怎样分配。三种问题的控制条件不同。本文的证据重点在于最后一种：在保持输入和响应相同的情况下，将尖峰交回数值模型后类型综合指标提高。这使结果可以归因于已定义的输出规则，但不意味着同样分工在其他模型、采样频率或异常密度下必然成立。\n"
    types += "\n两类点的差异说明，标签粒度与证据范围需要分别考虑。两点均数值变体到主方法的主指标提高0.0800，对应阶跃F1从0提高至0.3200后按四类等权；统一排名融合到主方法提高约0.0601，则来自尖峰误报减少。两条比较使用相同区间规则，分别检验阶跃的视觉补充与尖峰的数值保留，形成比单纯比较有无视觉更具体的解释。\n\n"
    types += case_intro + "\n\n" + transient_marker + transient
    types += "\n同一固定选样规则下" + failures

    ablation = "信息消融分别改变同一形态分工方法的输入。去局部图同时移除正常参考特写和全部六幅候选图，保留全局图及文字候选；去显式数值建议移除分类建议、分数及点邻域数值，但检查窗仍由数值模型选择。后者仍含数值参与，不能与纯视觉混同。\n\n" + ablation.lstrip()
    ablation += "\n这两种消融没有支持“信息越多越好”的单调关系。完整输入可改善某些进入—返回形态，却也可能使其他类型更难判断；因此，局部图应视为可检验的输入选择。由于各输入分别产生视觉响应，当前差异还包含单次生成波动，不能据一次消融确认固定的内部机制。结构简化的结论限于本批观测到的总体收益与错误变化。\n"

    boundaries += "\n固定边界比较进一步区分活动是否被确认与活动范围是否完整。数值和视觉只需存在有效日交集即可提供同类支持，这一支持不能保证整个视觉区间的首尾正确。视觉边界减少停止后假日而增加漏日，反映的是更保守的活动范围，而非所有边界错误同时减少。\n\n"
    boundaries += deformation_marker + deformation

    combined_uncertainty = seeds + "\n" + uncertainty
    combined_uncertainty += "\n因此，本实验可以支持主方法相对所列数值对照的条件性改善，以及同响应形态分工带来的可解释增量；相对其他视觉融合规则的总体优势仍受到区间跨零和事件数量的限制。后续结论不以选择另一个较高F1来消除这些限制，而应把类型均衡、异常覆盖与活动边界分别报告。\n"

    new_results = "# 4 结果与分析\n\n" + (WORKING / "results-opening-revision.md").read_text(encoding="utf-8").strip() + "\n\n"
    sections = [
        ("4.3 整体识别性能", overall),
        ("4.4 类型差异与代表序列", types),
        ("4.5 信息消融与结构简化", ablation),
        ("4.6 持续活动边界的误差取舍", boundaries),
        ("4.7 跨种子表现与结果不确定性", combined_uncertainty),
    ]
    for heading, body in sections:
        new_results += f"## {heading}\n\n{body.strip()}\n\n"
    (CHAPTERS / "04-results.md").write_text(new_results.rstrip() + "\n", encoding="utf-8")
    (CHAPTERS / "05-discussion.md").write_text(
        (WORKING / "discussion-revision.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    conclusion = (CHAPTERS / "07-conclusion.md").read_text(encoding="utf-8").replace("# 7 结论", "# 6 结论", 1)
    (CHAPTERS / "06-conclusion.md").write_text(conclusion, encoding="utf-8")

    supplement = ROOT / "supplement"
    supplement.mkdir(exist_ok=True)
    old_experiments = (CHAPTERS / "04-experiments.md").read_text(encoding="utf-8")
    supplement_text = "# EXP04复现与比较细节\n\n本材料承接结构优化稿未在主文展开的运行参数、方法映射与评价细节；实验记录未改变。以下来自初稿的已核查说明，标题编号仅作材料内部定位，不属于结构优化稿章节。正文中的数据及关键评价口径仍保留。引用键对应 manuscript/working/extra-references.json，完整源码及原始数据入口见仓库README。\n\n"
    supplement_text += old_experiments.partition("\n")[2].strip() + "\n"
    (supplement / "reproducibility.md").write_text(supplement_text, encoding="utf-8")
    for name in ["04-experiments.md", "05-results.md", "06-discussion.md", "07-conclusion.md"]:
        (CHAPTERS / name).unlink()


if __name__ == "__main__":
    main()
