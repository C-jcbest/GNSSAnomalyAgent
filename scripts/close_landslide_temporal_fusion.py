"""Record the stopped fusion candidate, actual visual review, and LaTeX manuscript."""

from __future__ import annotations

import re
from pathlib import Path

from prepare_landslide_ai_reference_holdout import verify_manifest
from probe_landslide_temporal_fusion import verify
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
RUN = BASE / "temporal-fusion-development-v1"
ANALYSIS = BASE / "temporal-fusion-development-analysis-v1"
REPORT = BASE / "temporal-fusion-development-report-v1"
INSPECTION = BASE / "temporal-fusion-developer-inspection-v1"
LITERATURE = BASE / "temporal-fusion-literature-v1"
OUT = BASE / "manuscript-v38"

FORMULATION = r"""本候选把已有日XGBoost的连续运动证据与同遍全局VLM区间放在共同日历上解码，减少自由区间修订对活动内部的整段覆盖。它使用标准二状态时间解码，不以融合或动态规划本身主张新颖性。记原完整观测掩码为 $O_t$，数值特征支撑为 $N_t$，数值概率输出为 $p_t$，已解析全局视觉状态为 $v_t\in\{-1,0,1\}$。定义

$$
e_t=\mathbf{1}_{\{O_tN_t=1\}}\left[\operatorname{logit}(p_t)-\operatorname{logit}(\theta)\right]
+\alpha\mathbf{1}_{\{O_t=1,\ v_t\ge0\}}(2v_t-1).
$$

实现仅为有限算术将概率夹到 $[10^{-6},1-10^{-6}]$；视觉活动/静稳是带符号硬投票，未知贡献零，不虚构VLM概率。数值概率及其log odds也未做现场校准，不能写成物理置信度。对每个有至少一方证据的连续已观测链 $C$ 求

$$
\widehat{y}_C\in\arg\min_{y_t\in\{0,1\}}
\left\{\sum_{t\in C}\frac{(1-2y_t)e_t}{2}
+\lambda\sum_{\substack{t,t-1\in C}}\mathbf{1}_{\{y_t\ne y_{t-1}\}}\right\}.
$$

缺测或双方均无支撑日拆链并保留未知；数值无支撑但视觉有判断的日期仍可由视觉决策，未当成原始缺测。用前向/后向最小代价求每一天两状态的全局最优成本，算术容差 $10^{-12}$ 内相同则留未知。此弃判仅处理等价最优解，不是经过校准的不确定性。固定切换罚 $\lambda=7\ln2$：在示意证据odds为2时，内部14日运动抵消两次切换代价；它不是最短持续时间保证或物理阈值，未扫描持续期/罚项。全局双向解码属于离线回顾。

复用两参考目标原6训练/6验证记录和已冻结XGBoost，不重新拟合；只新增6验证记录×2遍的原全局提示，共12次调用。开发0157—0168的两遍直接视觉回答全部精确复用。联合融合与去连续性各在原验证比较三权重.5/1/2，数值阈值沿用原冻结值；数值逐日和数值解码对照各独立比较原三阈值.35/.5/.65。按两遍平均日F1＋事件F1选择，共24试次。AI目标联合/去连续性选权重.5及阈值.35，数值解码选.5、数值逐日选.35；过程目标联合选权重.5、阈值.65，去连续性选权重2、阈值.65，两数值对照均选.35。所有选择在生成开发预测前冻结。去数值保留同一解码及视觉权重；阶段仍共同冻结数值头，不作为视觉阶段创新。

继续条件生成前登记：联合方法每遍F1均高于独立验证数值解码，平均弱R/事件F1均不降低，且对去连续性和直接视觉的平均F1与净正确均严格正。任一失败即停止，不改权重、罚项、提示、重复或子集救分。
"""

FINDINGS = """主方法AI活动F1为.8468/.8042，相对直接视觉平均+.035163、净正确+258，相对去连续性平均+.004071、净正确+40。但相对独立验证数值解码，第一遍+.036152、第二遍−.006506，平均事件F1差−.056184；未通过预登记继续条件，决定STOPPED。去连续性及直接视觉的对照通过不能救主数值比较失败，也不把过程F1 .8401/.8217改成主结论。原始多尺度数值规则F1 .8552且0API，两遍均高于本候选；原XGBoost .8044、持续性XGBoost .7880、robust .7163、PELT .5688及固定/自适应视觉全部保留。

必须分清阈值变化、观察覆盖与视觉纠错。评分后只做不参与选择的同阈值诊断：在联合方法冻结阈值去掉视觉，AI目标5998个数值支撑已知日的视觉增量仅第一遍6得/2失（净4）、第二遍8得/2失（净6）。1315个数值无支撑已知日中，第一遍新增1210正确判断，其中1089为未知→静稳、121为未知→活动；第二遍1050，其中934/116分别为这两类。该范围仍新增59/214个阳性误报，并非可靠地恢复所有缺支撑运动。它不是新候选或额外调参，更不能将数值弃判后的所有未知→静稳称为异常检出改善。过程参考下同阈值支撑日第一遍净−30、第二遍净+23，显示证据增量也随目标变化。

当前Codex在评分后实际查看6条全局原始位移/预测对齐图及3张局部原始位移/数值分数图，未改参考或预测。0160两遍恢复[276,324)的48个确认活动日；原始N/E块中位数从[255,275)的.520/−2.458到[295,315)的1.836/−5.125、[340,360)的3.077/−6.116mm，与图中持续运动一致。但仍漏其余已确认活动日，不能称边界已修复。0163两遍恢复35个弱尾TP、撤25个早期FP；尾部E中位数在[520,550)/[550,580)/[580,610)/[610,640)为−66.618/−68.089/−69.481/−69.858mm，存在连续弱累积，而解码仍把不少尾日判静稳。此处550—579的20个实际观测日没有数值特征支撑，不能把特征支撑不足称原始缺测。

0164两遍丢24TP，其中[280,294)的14日全部有观测和数值支撑，但因负分数被判静稳，使原连续活动出现人为缺口；原始N中位数从[260,280)3.935到[280,294)5.537、[294,314)6.812mm，实际局部图仍向上累积。另丢[410,420)10TP，却撤11/16平台FP，呈现活动与平台取舍。0159第一遍增54TP、撤15FP，同时损2TP、增6FP；第二遍丢69个弱尾TP、撤59FP，净正确−10。尾部E在[530,560)/[570,600)/[600,630)/[630,660)中位数−58.120/−59.305/−59.770/−62.426mm，与局部图慢累积一致；其数值负分数不能作为真实静止证据。0161第二遍撤224平台FP、丢1TP及增6FP；全局图中两次位移之间的平台支持撤回，但数值无支撑边缘仍沿用错误视觉活动。0167在缺测邻域补回数值未知、相对直接视觉没有改变已知日，说明部分收益是利用已有视觉覆盖，不是新检出。

主方法AI事件F1(.3/.5)为.5641/.4615与.5000/.4091；确认内核起点MAE两遍32.636日，终点28.000/30.455日。活动内部A/S/D macro-F1 .8468/.8455、覆盖.9675/.9709，端到端阶段macro-F1 .7567/.7550；共同数值阶段头不支撑新增视觉阶段能力。两遍已知日活动覆盖均1.0，但1066个未知参考日被判阳性436/461、静稳630/605，不能据此确认这些边界正确。本轮是已暴露开发记录上的描述性机制试验，不报告独立观测准确率、显著性或现场泛化。

本候选停止；原已确认的有限冲突观察机制保留，默认识别未改。新诊断提示困难主要在数值支撑缺口和持续弱累积的分数错配；现有视觉硬投票在数值支撑日几乎没有增量，不能继续以标准平滑或更多融合权重宣称贡献。后续机制必须提供区别于既有持续性数值规则的视觉增量，并在未参与方法选择的新记录上冻结验证；当前开发者视觉/数值参考继续披露身份与历史暴露，不升级为独立专家。
"""


def seal(folder):
    write_json(
        folder / "manifest.json",
        {
            "files": {
                path.relative_to(folder).as_posix(): sha(path)
                for path in folder.rglob("*")
                if path.is_file()
            }
        },
    )


def close():
    verify(RUN)
    for folder in (ANALYSIS, REPORT):
        verify_manifest(folder)
    if read_json(RUN / "state.json")["status"] != "predictions_complete":
        raise ValueError("Only write a manuscript after terminal inference")
    decision = read_json(ANALYSIS / "decision.json")
    if decision["candidate"] != "STOPPED":
        raise ValueError("This conclusion requires the actual frozen stop decision")
    viewed = [
        f"temporal-fusion-development-report-v1/{cid}.png"
        for cid in (
            "case_0160",
            "case_0163",
            "case_0161",
            "case_0159",
            "case_0164",
            "case_0167",
        )
    ]
    viewed += [
        f"temporal-fusion-developer-inspection-v1/{name}.png"
        for name in (
            "case_0159-420-730",
            "case_0163-480-680",
            "case_0164-260-440",
        )
    ]
    inspection = {
        "reviewer": "Current Codex; method developer; prior predictions and references exposed",
        "independent_expert": False,
        "new_labels": False,
        "predictions_or_references_changed": False,
        "actually_viewed_images": viewed,
        "image_count_actually_viewed": len(viewed),
        "findings": FINDINGS,
        "numeric_auxiliary_file": str(INSPECTION / "raw-numeric-evidence.json"),
    }
    write_json(INSPECTION / "inspection.json", inspection)
    seal(INSPECTION)
    write_json(
        LITERATURE / "reading-scope.json",
        {
            "type": "Targeted search and method/source reading, not systematic review",
            "queries": read_json(LITERATURE / "queries.json"),
            "read": {
                "2605.23116v1": "Metadata/abstract and section3.3; derived text lines671-937",
                "2609.07455v1": "Metadata/abstract and III-C/III-D; derived text lines291-562",
                "corevad-README.md": "First95 lines: setup/data/inference, read in preceding segment",
                "corevad-eval_ucf.py": "Entire evaluation source, including process_video, Gaussian functions and main",
            },
            "corevad_author_repository": "https://github.com/Muk-00/CoReVAD",
            "corevad_commit": "76ea2a0ac4ee521b9ea8d28898ad4c12731e3ef0",
            "author_model_executed": False,
            "parser_free_source_confirmed": False,
            "whole_papers_read": False,
        },
    )
    literature = """# 连续运动融合候选的最近邻核查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，沿用已授权研究。
- Origin Date: 2026-10-11。
- Verification Status: 官方arXiv书目/摘要及登记方法段落已读，CoReVAD作者评估源码已读；未执行作者模型。
- Version Label: temporal-fusion-literature-v1。

两条官方arXiv查询返回8条元数据，属于定向检索，不是完整或系统综述。原字节、查询、作者仓库commit和具体阅读范围保留于本目录。下载全文不代表全文已读。

CoReVAD（Lim与Hur，arXiv:2605.23116v1）3.3已有VLM分段判断、视觉/文字相似度softmax重加权、Gaussian平滑与视频位置权重。作者eval_ucf.py的process_video实际读取cleaned responses，按首句生成硬0/1，再以CLIP相似度归一权重加权这些标签，段级卷积、扩帧及中心位置加权，最后算AUROC。PDF/HTML公式索引与代码必须分别核查，本记录以所读实际代码说明执行行为。它不是GNSS同任务基线；未下载模型/数据或执行作者推理，不将论文分数移填GNSS表。

Parser-Free VLM Verification（Thuau、Gran、Haidar与Chelouah，arXiv:2609.07455v1）III-C/III-D已有数值MIL选段、冻结VLM复核、分数替换/线性融合和时间处理。其主接口提取Yes/No下一token logits的二元softmax作为排序信号，作者也不把它称校准概率；文本接口存在解析/平滑敏感性。本文仅硬区间投票，没有获取这些logits，不能将投票冒称该接口复现。未确认作者源码。

因此数值/VLM融合、选择性验证、硬标签与时间平滑均有先例；本GNSS候选仅检验原日历、缺测/支撑处理及持续运动解码是否产生额外识别价值。实际对强数值继续门失败，不能以标准动态规划包装贡献。
"""
    (LITERATURE / "notes.md").write_text(literature, encoding="utf-8", newline="\n")
    seal(LITERATURE)
    OUT.mkdir(exist_ok=False)
    manuscript = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    original = manuscript.read_bytes()
    (OUT / "manuscript-v37-original.md").write_bytes(original)
    old = original.decode("utf-8").replace("\r\n", "\n")
    if "> 完整中文初稿v37" not in old:
        raise ValueError("Expected the exact previous manuscript version")
    table = (REPORT / "compact-table.md").read_text(encoding="utf-8")
    cost = read_json(ANALYSIS / "cost.json")
    previous = read_json(BASE / "manuscript-v37/run-ledger-end.json")["cumulative"]
    cumulative = {
        "launches": previous["launches"] + cost["new_launches"],
        "receipts": previous["receipts"] + cost["new_receipts"],
        "at_least_reported_tokens": previous["at_least_reported_tokens"] + cost["reported_tokens"],
    }
    verification = read_json(REPORT / "verification.json")
    execution = (
        f"新增{cost['new_launches']}验证启动/{cost['new_receipts']}回执、{cost['reported_tokens']}报告token、"
        f"{cost['seconds']:.3f}秒，{cost['failures']}失败、0重试；开发集0新增调用、0新记录、0新拟合。"
        "单遍部署每条全局调用一次，两遍各12请求、29680/29692token，数值对照0API。"
        "这不是新增局部Agent复核，也不能把共享旧回答算作零部署成本。"
        f"两目标各110方法，共220评分；{verification['unchanged_baseline_target_rows']}旧评分/成本行全部字段保持。"
        f"{verification['exact_replay_files']}评分文件字节精确重放，0新增远端调用；"
        "5项短序列穷举/缺测/支撑业务测试与Ruff检查通过。"
        "12对齐图及3局部诊断图导出，实际阅读9张，未冒称全部读图。"
        f"累计{cumulative['launches']}启动/{cumulative['receipts']}回执/至少{cumulative['at_least_reported_tokens']}报告token，"
        "受控168/半合成60保持。\n\n"
    )
    section = (
        "### 6.20 连续运动分数与视觉区间时间解码\n\n"
        + FORMULATION
        + "\n"
        + table
        + "\n\n"
        + FINDINGS
        + "\n"
        + execution
    )
    section += (
        "[统一比较、逐日损益、同阈值诊断及视觉复核](landslide-temporal-fusion-results.md)。\n\n"
    )
    text = old.replace("## 7 讨论", section + "## 7 讨论", 1)
    text = re.sub(
        r"^> 完整中文初稿v37.*$",
        "> 完整中文初稿v38，2026-10-11。新增连续运动分数/VLM时间解码、110方法×2目标比较、消融与实际视觉/数值复核，候选STOPPED。公式保留Markdown LaTeX；作者AI参考不属于独立专家或现场金标准。本稿不是投稿定稿。",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    nearest = "\nCoReVAD[18]的作者代码已有VLM硬判断、相似度重加权和Gaussian时间平滑；Parser-Free VLM Verification[19]已有数值MIL选择VLM复核及分数融合。已分别读取3.3和III-C/III-D及CoReVAD评估代码，未执行作者模型。本文不以融合/解码本身主张创新，连续运动候选须证明对强数值的视觉增量，当前试验未通过。\n\n"
    text = text.replace("## 3 任务定义", nearest + "## 3 任务定义", 1)
    references = "[18] Lim, H., & Hur, Y. (2026). CoReVAD: A Contextual Reasoning Framework for Training-Free Video Anomaly Detection. [arXiv:2605.23116v1](https://arxiv.org/abs/2605.23116v1). 第3.3节与[作者评估源码](https://github.com/Muk-00/CoReVAD/blob/76ea2a0ac4ee521b9ea8d28898ad4c12731e3ef0/src/ucf/eval_ucf.py)已读，未运行作者模型。\n\n[19] Thuau, S., Gran, A., Haidar, S., & Chelouah, R. (2026). Parser-Free VLM Verification for Federated Weakly Supervised Video Anomaly Detection. [arXiv:2609.07455v1](https://arxiv.org/abs/2609.07455v1). III-C/III-D已读，未确认作者源码或运行作者模型。\n\n"
    text = text.replace("## 补充结果与完整数据", references + "## 补充结果与完整数据", 1)
    formulas = re.findall(r"\$\$[\s\S]*?\$\$|(?<![\\$])\$(?!\$)[^$\n]+\$(?!\$)", old)
    tables = re.findall(r"(?:^\|.*\n)+", old, re.MULTILINE)
    if not all(value in text for value in formulas + tables):
        raise ValueError("Every previous LaTeX formula and result table must remain exact")
    manuscript.write_text(text, encoding="utf-8", newline="\n")
    result = (ANALYSIS / "report.md").read_text(encoding="utf-8")
    result += (
        "\n## 方法与实际视觉/数值复核\n\n"
        + FORMULATION
        + "\n"
        + table
        + "\n\n"
        + FINDINGS
        + "\n"
        + execution
    )
    result += "## 实际读图清单\n\n" + "\n".join("- `" + name + "`" for name in viewed) + "\n\n"
    result += "## 离线复现\n\n完整推理拒绝重启，只重放冻结回答与数值；目标目录须不存在。\n\n```powershell\n$env:PYTHONPATH='src;scripts'\n& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/probe_landslide_temporal_fusion.py score --out artifacts/landslide-adaptive-review-2026-10-09/temporal-fusion-development-v1 --analysis artifacts/landslide-adaptive-review-2026-10-09/temporal-fusion-extra-offline-replay\n```\n"
    (ROOT / "docs/landslide-temporal-fusion-results.md").write_text(
        result, encoding="utf-8", newline="\n"
    )
    status = (
        "## 2026-10-11：连续运动时间解码停止，初稿v38\n\n[完整结果](landslide-temporal-fusion-results.md)。\n\n"
        + FINDINGS
        + "\n"
        + execution
    )
    status += "精确保留v37源文件及全部旧公式/结果表，新增LaTeX证据与解码目标式。研究goal active；不追加本候选的开发集提示/权重搜索。下方为历史记录。\n\n"
    for name in (
        "project-status.md",
        "landslide-adaptive-review-method.md",
        "landslide-submission-roadmap.md",
    ):
        path = ROOT / "docs" / name
        title, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(
            title + "\n\n" + status + remainder.lstrip("\n"), encoding="utf-8", newline="\n"
        )
    write_json(
        OUT / "content-check.json",
        {
            "old_formulas_preserved": len(formulas),
            "old_tables_preserved": len(tables),
            "old_manuscript_sha256": sha(OUT / "manuscript-v37-original.md"),
            "new_manuscript_sha256": sha(manuscript),
            "candidate": decision["candidate"],
            "default_changed": False,
        },
    )
    write_json(
        OUT / "run-ledger-end.json",
        {
            "goal_status": "active",
            "turn_classification": "progress",
            "cumulative": cumulative,
            "new_launches": cost["new_launches"],
            "new_records": 0,
            "new_training": False,
            "candidate": decision["candidate"],
            "next": "Do not rescue this candidate; test distinct visual motion evidence only with numeric and coverage controls, then freeze unseen confirmation records.",
        },
    )
    print(read_json(OUT / "content-check.json"))


if __name__ == "__main__":
    close()
