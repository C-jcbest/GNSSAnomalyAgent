"""Write a focused manuscript around the unchanged method and its fresh confirmation."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from close_landslide_utility_round import comparison_table, read_rows
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def close():
    run = BASE / "cross-seed-confirmation-v1"
    analysis = run / "analysis-v1"
    draft = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous = draft.read_bytes()
    if previous != (BASE / "manuscript-v19/manuscript-v19.md").read_bytes():
        raise ValueError("Current manuscript is not delivered v19")
    decision = read_json(analysis / "decision.json")
    rows = read_rows(analysis / "summary.csv")
    indexed = {row["method"]: row for row in rows}
    changes = read_json(analysis / "changes.json")
    costs = read_json(analysis / "actual-costs.json")
    old = read_json(BASE / "manuscript-v19/run-ledger-end.json")
    ledger = {
        "new_requests_started": costs["new_started"], "new_completed_local_receipts": costs["new_local_receipts"],
        "new_valid_model_json": costs["new_returned_json"], "new_reported_tokens": costs["new_reported_tokens"],
        "cumulative_requests_started": old["cumulative_requests_started"] + costs["new_started"],
        "cumulative_local_receipts": old["cumulative_local_receipts"] + costs["new_local_receipts"],
        "cumulative_reported_tokens_lower_bound": old["cumulative_reported_tokens_lower_bound"] + costs["new_reported_tokens"],
        "new_records": 12, "pure_controlled_records": 120, "observed_residual_semisynthetic_records": 60,
        "inference_retries": 0, "failures": costs["failures"], "goal_status": "active", "round_status": "progress",
        "scope_confirmed_on_new_seed": decision["scope_confirmed"],
        "method_confirmed_on_new_seed": decision["method_confirmed"], "field_accuracy_confirmed": False,
        "billing": "Lower bound of reported usage only; historical receipt and usage gaps retained",
    }
    out = BASE / "manuscript-v20"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    (out / "manuscript-v19-original.md").write_bytes(previous)
    selected = [indexed[name] for rep in (0, 1) for name in (
        f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target",
        f"adaptive_no_feedback_r{rep}__whole_target", f"adaptive_no_feedback_r{rep}__conflict_scope",
        f"adaptive_no_feedback_r{rep}__matched_center_scope", f"adaptive_numeric_feedback_r{rep}__conflict_scope")]
    selected += [indexed[name] for name in ("robust_rule", "pelt_rule", "xgboost",
        "raw_multiscale_rule", "xgboost_raw_persistence", "xgboost_42_matched_support", "xgboost_84")]
    evidence = []
    for rep in (0, 1):
        method = f"adaptive_no_feedback_r{rep}__conflict_scope"
        for control, records in changes[method].items():
            totals = {key: sum(row[key] for row in records.values()) for key in (
                "gained_correct", "lost_correct", "net_correct", "gained_tp", "lost_tp", "added_fp", "removed_fp")}
            evidence.append({"method": method, "control": control, **totals,
                             "positive_records": sum(row["net_correct"] > 0 for row in records.values())})
    save_json(out / "reported-correction-totals.json", evidence)
    save_json(out / "run-ledger-end.json", ledger)
    save_json(run / "run-ledger-end.json", ledger)
    section = f"""## 新随机种子的冻结确认结果

新seed2026101073、case0109—0120，六家族各两条、730日；输入存在前冻结原raw21/42冲突、14日数触发、最多两180日目标、49日权限、原图/提示/阶段头及九项强数值模型。训练/验证仍为原受控train24/validation12，分布与校准机会不变；没有新增模块、数值投票、检索、参数搜索或使用新标签训练。新参考为已知潜在活动/阶段，只支持受控恢复，未建立领域可观测或现场标签。

{comparison_table(selected)}

主F1 .8163/.7581，高于直接.7413/.7315和固定.7073/.7127；弱R .6745/.5668高于直接两遍.5025。对直接净确定纠错+193/+80，各6/3记录受益；新增389/184TP、损失129/75TP、新增164/100FP、撤176/75FP，说明总收益不等于逐日无损。两遍整个方法登记条件通过。XGB84 F1 .9043、弱R .7735更高，42同支撑XGB F1 .8915；强数值优势完整保留。

范围第一遍相对整目标撤23FP，无TP损失，F1增加.00417；净确定纠错+60，其中另37个未知改正确阴性。第二遍FP/TP/日F1均不变，但12个未知改正确阴性，净+12，覆盖提高。两遍均无损不等于满足登记的两遍各减FP条件；第二遍该项失败，范围额外收益整体未确认。等日数居中日F1 .8240/.7666更高，分别损失3/1个整目标TP；它不能被省略，也不换成登记主方法。主与直接的整记录描述性95%区间两遍跨零，不主张统计显著性。

主事件F1 IoU.3/.5为.7500/.7500与.7586/.7586，高于直接.6897/.6207、.6667/.6000及XGB84 .6316/.5263。主匹配起点MAE24.42/28.18日、终点43.58/41.09日，并非所有边界更好。活动内ASD宏F1 .7525/.7362、端到端.7057/.6758，均低于XGB84 .7792/.7558；阶段来自共同数值头，不称原生视觉阶段创新。29配置、117真实payload、108数值记录预测重算、254040逐日行及全家族结果保存。

实际新增117启动/117回执/117成功JSON、463439token，无失败/重试；主部署每遍26请求、86736/86766token，固定36/127432或127514，直接12/29730或29735。比固定少27.8%调用；相对直接增加14次局部请求，范围掩码不额外调用。数值CPU、渲染与范围CPU未单独计时，不填零CPU。累计{ledger['cumulative_requests_started']}启动/{ledger['cumulative_local_receipts']}回执/至少{ledger['cumulative_reported_tokens_lower_bound']}token。新12条受控参考与既有60半合成分开。

![新种子同方法比较](../artifacts/landslide-adaptive-review-2026-10-09/cross-seed-confirmation-v1/analysis-v1/cross-seed-comparison.png)

"""
    report = "# 原冲突范围方法跨种子确认：整体视觉收益复制，范围严格条件未全部复制\n\n" + section
    report += "\n## 全部29配置\n\n" + comparison_table(rows)
    report += """

## 复现

本轮117请求已经结束，禁止重跑或覆盖。仅分析重放使用不存在的输出目录：

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_cross_seed_confirmation.py --run artifacts/landslide-adaptive-review-2026-10-09/cross-seed-confirmation-v1 --scoped artifacts/landslide-adaptive-review-2026-10-09/cross-seed-confirmation-v1/scope-v1 --out artifacts/landslide-adaptive-review-2026-10-09/cross-seed-score-replay
```

cross-seed-confirmation-data-v1/pre-generation-freeze.json冻结生成前源/方法/评分器/模型/条件，manifest.json冻结输入和参考；run的freeze.json/prediction-freeze.json及scope-v1绑定实际图片、全部回答与修订。参考只在完整回答冻结后用于评分。原v19精确字节、完整历史负结果与旧源/预测不改。全记录活动/阶段图由原runner生成，不择最好重复。
"""
    (ROOT / "docs/landslide-cross-seed-confirmation-results.md").write_text(report, encoding="utf-8")
    (out / "cross-seed-report.md").write_text(report, encoding="utf-8")
    old_text = previous.decode("utf-8").replace("\r\n", "\n")
    methods = old_text.split("## 3 任务定义", 1)[1].split("## 5 实验设计", 1)[0]
    methods = "## 3 任务定义" + methods
    methods = methods.replace("局部结果仅更新目标窗内；窗外保留全局判断。",
        "整目标对照仅更新目标窗内；主方法进一步遵守第4.5节的冲突范围权限，范围外保留全局判断。", 1)
    methods = methods.replace("它是新的实验方法，历史整目标默认和其冻结实现不被追改。",
        "本稿登记主方法包含此权限；新seed确认保持该源码原字节，历史整目标对照及所有失败版本不被追改。", 1)
    references = {}
    for line in old_text.splitlines():
        match = re.match(r"^\[(\d+)\] ", line)
        if match:
            references[int(match.group(1))] = line
    chosen_references = list(range(1, 9)) + [13, 22, 23]
    reference_text = "\n\n".join(re.sub(r"^\[\d+\]", f"[{index}]", references[old_index])
                                    for index, old_index in enumerate(chosen_references, 1))
    manuscript = f"""# 原始持续运动证据驱动的有限视觉复核：长期GNSS活动识别的改进与适用边界

> 完整中文初稿v20，2026-10-10。围绕一个已实现方法重写正文，历史探索留在补充报告与精确v19快照；新随机种子整体视觉收益复制，范围严格条件未全部复制。受控机制参考与现场可观测参考分开，本稿不是投稿定稿。

## 摘要

长期GNSS累计位移中的弱持续运动可能被较强历史运动压缩，高位平台又容易被视觉模型误报为持续活动。本文提出原始持续运动证据驱动的有限视觉复核：以三个不相交原始位移中位数块的连续同向变化，与全局视觉初判构成逐日冲突，选择至多两个局部目标；局部VLM提供活动区间，数值证据仅决定观察位置及可修订范围，不直接赋活动标签。活动外禁止A/S/D，缺测和未知保留。与直接视觉、固定观察、整目标改写、等可改写日数居中范围及数值文字反馈比较，并保留共同训练/验证的规则、PELT和强XGBoost控制。另一个生成前冻结随机种子的12条受控记录中，两遍主活动F1为.8163/.7581，高于直接.7413/.7315与固定.7073/.7127，弱召回提高，净确定纠错+193/+80，部署26次调用低于固定36次。XGBoost84的日F1 .9043仍更高，主方法的事件F1 .7500/.7586较高。范围第一遍撤23FP且无TP损失，第二遍未减少FP，未满足两遍均减少FP的严格条件；居中消融日F1更高。观测残差半合成的四未见背景上，主弱召回提高但整体F1下降，额外参照和校准分支未取得稳定收益。结果支持一个场景受限的视觉识别改进与可追溯修订机制，不能推出整体优于强数值、范围普遍最优、原生VLM阶段改善或现场准确率。

**关键词：** GNSS；持续位移活动；视觉模型；自适应观察；有限修订；弱运动

## Abstract

Weak sustained motion can be obscured by larger historical movements in long GNSS displacement plots. We propose bounded visual review selected by disagreement between a global visual prediction and two successive, direction-consistent changes in three disjoint raw median blocks. At most two local views are acquired; the VLM supplies activity intervals, while numerical evidence determines observation locations and revision authority. Missing observations and unknown decisions are retained, and A/S/D stages are gated by activity. On twelve newly generated, pre-frozen controlled records from another random seed, activity F1 reaches 0.8163/0.7581 versus 0.7413/0.7315 for direct vision and 0.7073/0.7127 for fixed review. Deployment requires 26 calls per repetition versus 36 for fixed review. XGBoost84 has higher daily F1 (0.9043), whereas the proposed method has higher event F1. Scoped revision removes 23 false-positive days without losing true positives in the first repetition, but does not reduce false positives in the second; its registered strict condition is not fully replicated, and the matched central-scope control has higher daily F1. Tests on observed-residual semisynthetic backgrounds reveal substantial false positives despite improved weak-motion recall. The results support a limited visual detection improvement and disclose its costs and failure modes, without establishing field accuracy or improved native visual stage reasoning.

## 1 引言

累计位移是否仍在变化，与其相对初值有多大偏移是两个不同问题。长期滑坡可表现为慢滑、间歇活动及季节响应[1]；GNSS还包含时间相关误差、尖峰和缺测[2]。直接视觉识别可利用整体形态，却可能漏掉大位移后的弱尾，或把保留累计偏移的平台继续赋活动。单靠导数提前响应或数值触发不能确定真实活动边界。

本文研究一个简单问题：由原始持续变化与初判冲突决定有限局部观察，能否比固定观察更有效地修正活动识别？进一步检验局部回答是否应获得整个目标的改写权限。方法不引入多Agent、强化学习或额外工具链；研究贡献在GNSS持续运动语义下的证据选择和结果修订。技术差异、受控收益、强数值优势以及背景迁移失败均报告。阶段评价用于检查活动改善的传递，不将共同数值阶段头称为视觉推理创新。

## 2 相关工作与技术差异

### 2.1 GNSS运动、变点和学习控制

滑坡季节慢滑的位移与孔压观测为持续运动任务提供物理背景[1]，但论文中的信噪筛选不是本项目活动真值。GNSS误差研究说明白噪声和相关噪声的重要性[2]；本仿真未经实际场地误差标定。PELT是成熟变点方法[7]，单阶跃同样会形成变点，需要结合段内运动。XGBoost[8]及原始多尺度规则得到共同训练/验证机会和与新方法相同的长尺度运动信息，构成强对手。

### 2.2 视觉与Agent方法

TAMA已有正常图示例和多尺度复核[3]；AnomaMind已有工具增强定位、确认/拒绝/细化及有限返回[4]。TSAD-Agents[5]及Refining[6]也涉及图/文和传统工具复核。因此观察、再次询问或工具框架本身不构成本文创新。

已核查TAMA官方commit db99a75a83d0ef05caf7ff0d199b75b594e98584：所读多尺度路径在初判非空时触发首尾重叠窗口；本方法也能由原始运动证据提出初判空区的观察需求。已核查AnomaMind的Locator/Actor/Detector/workflow/finalize，通用有限修订已有先例。本文具体区别是连续原始位移冲突同时控制GNSS观察位置和逐日权限；未据所读源码主张首创，也未把共同Qwen替身当作者SFT/RL模型复现。既有同任务TAMA适配结果保留在补充报告，本新种子批次未追加TAMA请求。

Anlu以参照条件化和反事实监督学习正常性[10]，MORA以同目标短/长上下文重建差修正异常分数[11]。本研究检验的正常实例检索与新增上下文并未取得稳定纠错，不能用不同参照图或单向规则称新贡献。两文只阅读登记方法段落，未运行作者模型或核实其源码；定向阅读范围和失败访问记录可追溯。

{methods}
## 5 实验设计

### 5.1 参考与数据隔离

历史非独立AI辅助开发标签只用于探索。受控生成器保留既有噪声、尖峰和缺测，替换为可追溯的静稳、仪器阶跃、弱恒速、A/S/D、再活动及强后弱运动。弱恒速.04—.09 mm/day、强活动峰速.4—.9 mm/day，活动起点和速率按既定随机流变化。参考活动先由潜在非零位移速度确定，再在其内部赋阶段；缺测掩码统一。该标签支持设计运动恢复，不能替代领域可辨识活动参考。

原seed2026100937的train24/validation12用于全部受控数值拟合和校准；首批机制12用于方法选择。case0049—0060确认简化纯图方法，case0073—0084确认49日范围。其他已见批次用于尺度、显示、阶段等探索，不能反复称独立确认。当前新seed2026101073、case0109—0120、730日、六家族各两条，在输入存在前冻结原方法、评分器、九控制模型和判定条件；新12标签未用于拟合或选参，同分布控制沿用适用的训练权重。

观测残差半合成另用官方P227 IGS20三分量[9]的十个不相交730日背景块，仅按实际日期逐轴OLS去线性趋势，保留季节变化、尖峰和缺测。每块六注入变体，共60条；背景组train0/1/6/7、validation2/3、test4/5/8/9为24/12/24记录，按背景隔离并重新拟合/校准全部强基线及阶段头。已知注入参考只评价额外运动恢复，未证明背景本身无真实运动。四test背景不是24独立站点；当前纯观测复核包无人完成，正式现场参考未知。

### 5.2 公平对比与最少消融

直接视觉只看全局图；固定组在日历三等分附近选择两个既有180日目标；自适应组按原始冲突选窗。匹配目标共享同次纯图回答，全局初判在同遍共享。整目标、冲突范围和每目标等可改写有观测日数的居中范围比较权限；数值文字组保持同窗同图，仅增加原始持续变化摘要。默认主组不含数值文字，不能按新分数更换主方法。

共同九数值控制为robust、PELT、原XGB、robust活动接学习阶段、多尺度raw42规则、raw42 XGB、raw84规则、42同支撑XGB及84 XGB。原robust/PELT各九候选，XGB六候选；新增信息规则各九、学习模型各六，按验证日F1加事件F1选参，不使用新评价标签。42同支撑与84学习组用相同有观测short/42/84支撑人口，以区分新增信息和更多可评价支撑的效果。新批无需重训同分布模型；半合成背景变化时已按共同新划分重训。

模型为qwen3.8-flash，temperature .1、top_p .3、max_tokens2048、关闭thinking，两遍完整公开、无重试。每记录主方法最多1全局加2局部；各消融共享请求，部署成本不得跨组相加为实际采集成本。远端底层权重无独立版本证明。

### 5.3 指标与预先条件

主要评价日活动P/R/F1、FP/FN、弱召回、IoU .3/.5一对一事件匹配、匹配条件起止MAE、未知覆盖和成本。未知在真活动中计漏判，参考缺测不评分。端到端ASD宏F1计活动外假阶段，真活动内ASD仍保留活动漏检和阶段未知。两遍不合并为独立记录，2000次完整记录配对bootstrap只作描述性区间。

新种子在生成前登记：整个方法两遍F1高于直接和固定、弱R不低于直接、对直接净确定纠错为正且至少两记录受益；范围额外作用则要求两遍相对整目标无TP/弱R损失、各减少FP。未知改未知或误报改未知不等同新增正确阴性。任一条件失败后不扫描参数、选择最高消融或再挑种子救分。事件、边界、强数值与阶段权衡完整保留，不预设全部指标优胜。

## 6 实验结果

### 6.1 既有冻结受控确认

初次纯图方法确认F1 .7227/.8426高于直接.6762/.7172，但多尺度数值规则.8729更高。随后范围确认F1 .7965/.7971，整目标.7928/.7934，直接.7269/.6992；范围两遍各撤18FP、无TP损失，全部修正集中case0078一条强后弱记录。等日数居中损失24/22TP，包括弱日；主范围没有保证所有弱运动被观察或确认。该批多尺度规则.7887、追加同支撑XGB84 .8477，强学习优势及追加控制设计时序均保留。

### 6.2 新种子结果与强基线

{comparison_table(selected)}

新种子主F1 .8163/.7581，弱R .6745/.5668，整个方法两遍登记条件通过。对直接净确定纠错+193/+80，受益记录6/3；新TP和丢TP为389/129、184/75，FP新增/撤回为164/176、100/75。整体收益并非无损修改，第二遍FP总数反而较直接多25日。主事件F1 .7500/.7586高于直接和固定，但边界MAE并未全面下降。主与直接的日F1差描述性95%区间两遍跨零，不能以12条两遍成绩宣称显著普遍改善。

强XGB84日F1 .9043、弱R .7735和阶段效果更高；主事件F1 .7500/.7586高于其.6316，较好的事件重叠指标与更低日准确性并存。42同支撑XGB .8915也超过主组。结果支持有限视觉流程相对普通/固定视觉的受控改进，不能称替代强数值。

![新种子同方法比较](../artifacts/landslide-adaptive-review-2026-10-09/cross-seed-confirmation-v1/analysis-v1/cross-seed-comparison.png)

### 6.3 权限、位置与反馈消融

范围第一遍撤23FP，无TP损失，F1增加.00417；另37个局部未知恢复正确阴性，净确定纠错+60。第二遍仅12个未知恢复正确阴性，FP/TP/F1不变。第二遍未满足各减FP的登记要求，额外范围收益整体未确认，不能把增加覆盖重新改名为原严格条件通过。

居中同日数范围日F1 .8240/.7666高于冲突范围，两遍分别损失3/1个整目标TP；第一遍弱R也更高。它说明新批并未建立冲突位置的独有日F1优势，仍保持登记主组。数值文字范围.8547/.8039也较高，但旧确认及背景任务没有稳定反馈增量，不因本批更好切换主方法。主与整目标的真活动内阶段分数不变，范围仅减少活动外误判或未知，不能称修好A/S/D内容。

### 6.4 观测残差背景与停止的扩展

四未见P227背景的24注入记录中，原方法F1 .5829/.5623，低于直接.6722/.6330、固定及重新拟合XGB84 .7261；弱R .8083/.7321高于直接.6178/.3910。范围只撤1/2FP，约99%选中目标有观测日已有冲突权限，背景趋势仍可能被扩大成活动。强后弱家族有局部收益，不足以抵消整体误报；不能把注入外背景变化确定解释成真实坡体假活动。

开发探索另检验整段支撑、84日证据、块显示、方向、收益校准、新增360图、年度参照及实际训练正常实例。局部或单遍收益未稳定迁移时均停止，未叠加进主方法。最新正常实例检索两遍F1 .5511/.5683低于同提示重复原图.5691/.6015，净纠错−350/−179；数值初判加VLM补充首遍F1 .5400低于XGB84 .7553，新增121TP也增1173FP，因此未执行缺失第二遍请求。完整负结果、所有强基线和失败版本在补充报告，不以这些探索替代主方法独立确认。

### 6.5 阶段、未知与成本

新批主活动内ASD宏F1 .7525/.7362、端到端.7057/.6758；XGB84对应.7792/.7558。主第一遍覆盖.9835、第二遍.9818，直接两遍为1；活动日F1改善不代表更少弃判。VLM没有在本轮供应原生阶段，不能将共同数值头成绩归为视觉阶段创新。

每遍主26调用/86736或86766token，固定36/127432或127514，少27.8%调用；直接12/29730或29735，故相对直接增加14局部请求。范围不额外调用。实际采集117启动/117回执/117成功JSON、463439token，无失败/重试。九数值模型无VLM调用，其CPU与渲染/范围CPU未分别计时，不报告零计算成本。29配置、117实际payload、108数值记录预测重算、254040逐日行及全部家族导出可追溯。整个研究累计2519启动/2518回执/至少11000829已报告token，历史缺失费用保留。

## 7 讨论

主要收益来自有限局部观察在受控弱运动任务中的纠错价值，而非复杂Agent架构。持续证据使部分全局初判空区获得观察，原图中的弱趋势仍由VLM解释；调用少于固定观察。权限约束在部分情况下保护全局正确阴性，但新种子没有复制两遍减少FP的严格效果，居中控制又取得更高日F1，因此其独立创新增量和普适性仍需更强证据。

观察与注入任务存在根本边界：季节背景和设计缓变可在原始曲线中相似。无独立正常制度或领域参考时，新增图、正常实例、数值一致性和未知保留并不自动提高识别可靠性。观测残差负结果不能隐去，单向保留旧TP也不能避免FP累积。本文不建立总运动地质归因或失稳预测。

其他限制包括每家族两新记录的小样本、固定绘图尺度、远端模型版本、离线未来信息、原生阶段贡献缺失以及未完整复现最近邻发布权重。不同随机种子的受控确认加强了算法可追溯性，但不等于跨真实场地验证。现阶段应保留简单方法和明确场景范围，避免以更多提示、模块或最佳重复弥补负结果。

## 8 可复现性与AI使用说明

执行源、协议、输入、模型选择、实际发送图、提示、请求哈希、回答、预测及评价保存在artifacts/landslide-adaptive-review-2026-10-09。新数据pre-generation-freeze.json在生成前固定方法和评分器；推理冻结后才读参考。独立sklearn重算活动和阶段，核查缺测、阶段门控、事件、原始请求payload及九数值预测。复现承诺为固定输入/协议/评价可追溯，远端生成不承诺逐字节相同。

本稿将历史试验细节移入既有补充报告，精确v19字节保留在manuscript-v20/manuscript-v19-original.md，所有旧源码/预测/失败不覆盖。AI参与代码、实验分析和初稿写作；AI辅助开发标签、已知潜在机制及未复核现场数据分别披露。当前领域复核者未落实，纯观测包仍unreviewed；作者审核及正式现场结论待完成。

## 9 结论

本文形成了一个最多两次局部请求的GNSS视觉识别方法，用连续原始位移冲突选择观察并限制修订，活动与阶段语义分开。新随机种子两遍支持其相对直接和固定视觉的受控活动收益及调用取舍；强数值日F1仍更高，范围严格条件未全部复制，居中消融优势和背景迁移失败限制了贡献范围。当前证据支持场景限定的方法初稿，不能写成普遍可靠性、独立现场效果或原生VLM阶段改进。

## 参考文献

{reference_text}

## 补充结果与完整数据

- [新种子全部29配置及复现](landslide-cross-seed-confirmation-results.md)
- [原冲突范围开发与独立确认](landslide-revision-scope-results.md)
- [观测残差半合成的重训基线与迁移失败](landslide-background-benchmark-results.md)
- [实际正常实例检索及数值初判提前停止](landslide-normal-anchor-results.md)
- [最近邻公开源码阅读范围](landslide-nearest-method-code-review.md)
- [精确历史v19初稿](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v20/manuscript-v19-original.md)
"""
    draft.write_text(manuscript, encoding="utf-8")
    (out / "manuscript-v20.md").write_text(manuscript, encoding="utf-8")
    save_json(out / "checks.json", {
        "v19_exactly_preserved": sha(out / "manuscript-v19-original.md") == sha(BASE / "manuscript-v19/manuscript-v19.md"),
        "current_matches_v20": sha(draft) == sha(out / "manuscript-v20.md"),
        "method_confirmed": decision["method_confirmed"], "scope_confirmed": decision["scope_confirmed"],
        "field_accuracy_confirmed": False, "analysis_checks": read_json(analysis / "checks.json"),
        "reference_renumbering": {str(old_index): index for index, old_index in enumerate(chosen_references, 1)},
        "goal_status": "active",
    })
    entry = """## 2026-10-10：原方法新种子确认完成，聚焦全文v20

[新种子结果](landslide-cross-seed-confirmation-results.md)：保持原raw21/42选窗＋49权限，新seed2026101073、case0109—0120、六族各两条。方法/评分器/九强数值模型在生成前冻结，原train24/validation12不重训或新选参。两遍主F1 .8163/.7581对直接.7413/.7315及固定.7073/.7127，弱R .6745/.5668对直接两遍.5025，净确定纠错+193/+80、受益6/3记录，整个方法登记条件通过；仍损失129/75原TP，不称总体无损。XGB84日F1 .9043、弱R .7735更强；主事件F1 .75/.7586较高，边界/阶段取舍完整。

范围第一遍撤23FP、无TP损失，另37未知改正确阴性，净+60；第二遍只修好12未知、FP/TP/F1不变，未满足两遍各减FP，范围额外收益整体未确认。居中等日数F1 .8240/.7666更高、丢3/1TP；不得省略或事后换主组。对直接的整记录描述区间跨零，不主张显著性或现场效果。

117新请求全部成功/463439token、无重试；部署主26/26对固定36/36及直接12/12；29配置、117payload、108数值预测重算、254040逐日行及全家族结果保存。聚焦全文v20围绕一个简单原方法、统一强基线/消融/新确认和背景失败，历史细节留补充报告及精确v19原字节，不覆盖冻结源码/预测。累计2519启动/2518回执/至少11000829token，120受控/60半合成分开。新严格范围条件未全部复制，不在此批改参数或挑下一seed救分。所有本轮推理/评分结束，goal active；正式现场可信参考仍缺，用户暂无复核者，继续自主，不再索要或造专家标签。

"""
    for name in ("project-status.md", "landslide-research-continuation-prompt.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / name
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + entry + rest.lstrip("\n"), encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8").replace("- [正常实例检索与数值初判补充]",
        "- [原方法新种子冻结确认](docs/landslide-cross-seed-confirmation-results.md)：整体视觉收益复制，范围严格条件未全部复制；117请求、29配置、聚焦全文v20。\n- [正常实例检索与数值初判补充]", 1)
    readme.write_text(text, encoding="utf-8")
    print({"version": 20, "decision": decision, "ledger": ledger}, flush=True)


if __name__ == "__main__":
    close()
