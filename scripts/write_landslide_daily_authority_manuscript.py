"""Update the complete manuscript with frozen RCRA evidence and LaTeX formulas."""

from __future__ import annotations

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json


def write(root):
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    run = base / "daily-authority-expansion-v1"
    replay = base / "daily-authority-expansion-replay-v1"
    report = base / "daily-authority-report-v1"
    for folder in (replay, report):
        verify_manifest(folder, "freeze.json")
    verify_manifest(run, "prediction-freeze.json")
    verify_manifest(run / "analysis-v1", "freeze.json")
    if read_json(replay / "verification.json")["status"] != "VERIFIED":
        raise ValueError("Matching complete replay required")
    summary = read_json(run / "analysis-v1/summary.json")
    decision = read_json(run / "analysis-v1/decision.json")
    methods = ("visual_daily_partial_r0", "visual_daily_partial_r1", "direct_visual_r0", "direct_visual_r1",
               "fixed_visual_r0", "fixed_visual_r1", "adaptive_visual_r0", "adaptive_visual_r1",
               "query_motion_mass", "global_motion_mass", "xgb_raw84", "xgb_raw12",
               "robust_rule", "pelt_rule", "visual_numeric", "pure_visual")
    ablations = tuple(f"visual_{name}_r{rep}" for name in
                      ("daily_partial", "whole_partial", "neighborhood_partial", "matched_partial",
                       "daily_strict", "no_local_visual") for rep in range(2))
    text = (root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v11.md").read_text(encoding="utf-8")
    text = text.replace(text.splitlines()[0], "# GNSS形变异常的有界视觉复核：逐日修订权限与机制验证", 1)
    start = text.index("> 中文完整研究初稿")
    end = text.index("\n\n## Material Passport", start)
    text = text[:start] + (
        "> 中文完整研究初稿，2026-10-13，v12。整体形变异常为主，跳变／尖峰独立识别且可同现；"
        "本版新增RCRA逐日冲突授权、小试及48／24完整开发扩展。96×730记录规模不变、保留24内容未加载；"
        "候选减少误报但有召回与严格事件取舍，完整预设条件未全通过。AI原图复核为开发者分析，尚无独立领域参考或已确认稳定贡献。"
    ) + text[end:]
    text = text.replace("deformation-manuscript-20261013-v11", "deformation-manuscript-20261013-v12", 1)
    start = text.index("## 摘要\n\n") + len("## 摘要\n\n")
    end = text.index("\n\n**关键词", start)
    abstract = (
        "长期GNSS形变异常识别需要区分持续位移、运动后平台和仪器跳变。已有视觉复核可修正部分异常，"
        "但整窗覆盖也可能把原判断正确的日期改错。本文提出并检验RCRA（Raw-Conflict Revision Authority，"
        "逐日冲突授权复核）候选：组合全局视觉初判、原始位移的短长尺度运动证据和最多两次局部视觉观察，"
        "仅允许新回答修订具有实际原日证据冲突的日期，数值证据不直接赋予最终活动标签。"
        "首先在16训练／8已暴露验证记录上隔离权限机制，两遍日F1为0.8982／0.8971，"
        "均高于直接视觉、整窗、49日邻域和同日数居中权限控制。随后扩展至48训练／24已暴露验证记录，"
        "复用同划分重训的规则、PELT、XGBoost与视觉数值强对照。主日F1为0.7540／0.7418，"
        "相对直接视觉0.6728／0.6749，误报日由547／483降至194／188，事件IoU=0.3的F1"
        "由0.7500／0.7273提高至0.8485／0.8235。然而第一遍日F1略低于49日邻域0.7570；"
        "相对整窗减少180／218误报日的同时少检110／111正确活动日，更严格IoU=0.5事件指标亦受损。"
        "强数值分布方法F1为0.8442，仍优于候选。预设完整条件未全部通过，停止固定版本。"
        "本文保存统一强对比、匹配消融、逐族结果和精确重放，保留历史机制负结果；"
        "当前证据支持减少错误扩展的开发价值及召回取舍，尚不宣称独立现场效果或保证发表。"
    )
    text = text[:start] + abstract + text[end:]
    start = text.index("## Abstract\n\n") + len("## Abstract\n\n")
    end = text.index("\n\n## 1 引言", start)
    text = text[:start] + (
        "Long-term GNSS deformation detection must distinguish continuing displacement from post-motion plateaus and instrument offsets. "
        "We test Raw-Conflict Revision Authority (RCRA), combining a global visual prior, short/long-scale evidence from raw displacement, "
        "and at most two local observations. Conflicts authorize revisions on their original supported dates, while local visual answers supply decisions. "
        "A 16-training/8-exposed-validation pilot attains daily F1 of 0.8982/0.8971 over two repeats and passes its comparison conditions. "
        "Expansion to 48 training and 24 exposed validation records uses refitted numerical and visual-feature controls with shared scoring. "
        "RCRA attains F1 of 0.7540/0.7418 versus 0.6728/0.6749 for direct vision, reducing false-positive days from 547/483 to 194/188. "
        "Nevertheless, its first-repeat F1 is below the 49-day neighborhood control at 0.7570. "
        "Compared with whole-window review, it removes 180/218 false positives while detecting 110/111 fewer true activity days; stricter event matching also degrades. "
        "A strong numerical distribution model reaches F1 of 0.8442. The fixed expansion fails its continuation gate and is stopped. "
        "These findings establish a development tradeoff between preventing erroneous expansion and recovering activity, "
        "rather than an independently validated Agent/VLM advantage. All comparisons, ablations, requests and historical negative results are retained; "
        "reserved records remain unopened."
    ) + text[end:]
    text = text.replace(
        "本研究选择一个可隔离的训练干预：保持首末位移、原观测误差与缺测基本相同，仅改变期间是否持续运动，检验模型能否学习额外的持续性差异。",
        "本稿最新主机制检验局部VLM应获得哪些原始日期的修订权限：原位移与初判冲突不仅选择观察，还约束编辑。"
        "早期同净位移训练干预及其负结果作为历史探索保留，不替代最新完整开发比较。",
        1,
    )
    related = (
        "\nRCRA最新最近邻核查实际继续阅读VLM4TS的Anomaly Verification and Refinement及Prompt Design，"
        "和Refining的候选子集方法。VLM4TS允许新增，Refining只保留／排除已有候选而不能提高原检测器召回。"
        "所读VLM4TS源码commit 8ab8c16414eb2c1a861dfc3e76f458180035a879可见空候选跳过、全局图与失败返回原候选，"
        "没有运行作者模型，也不把论文文字直接等同全部代码路径。DQS[19]用DTW选oracle标签再校准阈值，"
        "并讨论误标；它不是VLM日期权限机制。通用两阶段、有限调用或选择性复核不是RCRA首创。"
        "源码、原文与实际阅读边界见daily-authority-session-v1/reading-notes.md。\n\n"
    )
    position = text.index("## 3 ")
    text = text[:position] + related + text[position:]
    method = r"""### 3.16 RCRA：逐日冲突授权复核

本轮只新检验原日权限，复用现有21／42日持续运动、选窗、partial弃权保留、共同14观测日解码和阶段头，不把已有机制再命名为创新。令全局初判为 $\hat y^{(0)}_t\in\{-1,0,1\}$，$M_t$ 为原短／长尺度至少一项满足固定0.03 mm/day及噪声比2的运动描述，$U_t$ 为原日观测且至少一尺度可用：

$$
C_t=U_t\land\left(M_t\neq\mathbf1[\hat y^{(0)}_t=1]\right),\qquad
S(W)=\sum_{t\in W}C_t,\qquad
A_t=C_t\land\mathbf1\!\left[t\in\bigcup_{W\in\mathcal W^\star}W\right].
$$

730日记录按180日窗口／90日步长观察，至少14冲突日才候选；按冲突数及起始日排序，选最多两个互不重叠窗口，之后一次性执行复核。全局＋局部原图提示与旧实验完全相同，局部不注入数值摘要，反馈体现在选窗及修订权限，不是新提示数量。令明确局部回答为 $\ell_t$：

$$
\tilde y_t=\begin{cases}
\ell_t,&A_t=1\text{ 且存在明确成功局部回答},\\
\hat y^{(0)}_t,&A_t=0\text{ 或成功复核全部弃权},\\
-1,&A_t=1\text{ 且失败或矛盾未解}.
\end{cases}
$$

数值证据仅授权，主方法不直接赋予 $M_t$ 为活动答案。失败与成功弃权不同，失败不能恢复确定初判。旧49日邻域及同可修改观测日数的居中权限为匹配控制；去局部视觉只在权限内数值翻转初判，因此仍使用全局VLM，不能称完整纯数值方法。

输出应用公共持续期解码并赋同48TRAIN阶段：

$$
\hat{\mathbf y}=\mathcal D_{14,7}(\tilde{\mathbf y},\mathbf o,\mathbf b),\qquad
\hat z_t=\begin{cases}
g_{\rm stage}(\mathbf f_t),&\hat y_t=1\land b^{\rm stage}_t=1,\\
0,&\hat y_t=0,\\
-1,&\text{其他}.
\end{cases}
$$

解码只允许跨≤7日原缺测，不填未知阳性。权限保证解码前修改范围；被切碎的短段可能在权限外删除，须公开：

$$
P_{\rm outside}=\sum_t(1-A_t)\mathbf1\!\left[\mathcal D_{14,7}(\tilde{\mathbf y})_t\neq
\mathcal D_{14,7}(\hat{\mathbf y}^{(0)})_t\right].
$$

该机制不保证权限是完整运动范围，也不保证冲突外日期静止；中位数证据使用周围观测，是回顾识别，不是在线预警。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    protocol = (
        "### 4.4 RCRA完整扩展的预设边界\n\n"
        "96×730即70080个原日历记录日；当前扩展视觉覆盖到全部24已暴露验证，"
        "没有增加16条独立新记录。原8回答复用，剩余16保留两遍新推理；共同48TRAIN阶段"
        "及18重训／合理校准强对照，不拿16TRAIN小试模型冒充公平完整基线。"
        "两遍各需日F1胜过直接／固定／旧自适应／整窗partial／49邻域／同日数／去局部视觉，"
        "净纠错正、IoU.3事件不损、纯干扰FP与混合召回不损于直接。"
        "任一失败停止固定版，不调阈值／提示／重复次数；强数值差距始终披露，"
        "并不要求全部指标胜过全部方法。先冻结输出再读取参考；24保留内容未加载。\n\n"
    )
    text = text.replace("## 5 实验结果", protocol + "## 5 实验结果", 1)
    result = f"""### 5.18 RCRA：开发小试正收益，完整扩展出现F1取舍

同16TRAIN／8已暴露VALIDATION小试，视觉主方法F1 .898216/.897059，高于直接.808470/.869565、整窗partial均.830769、49日邻域均.837209、同日数居中均.874227、去局部视觉均.881696；事件IoU.3均1，视觉两遍各5/5条件通过。数值初判模式相对强数值净损3/15日，停止、未扩展。64方法512输出精确重放后，才执行完整开发扩展；小试强数值.945529不能与完整24条强数值.844167混比。

以下统一48TRAIN／24已暴露验证，所有流程共享同阶段头／缺测／持续期规则，18强基线重训与原调参机会保留：

{score_table(summary, methods)}

主相对直接视觉误报547/483→194/188，Precision .6823/.7039→.8597/.8600，Recall仍仅.6714/.6522。新得34/37TP且撤回20/30TP，净纠错367/302。第一遍F1 .753963略低于整窗partial .754355及49邻域.756993；第二遍.741811高于两者.735405/.737977。两遍各4/5及5/5，判定{decision['status']}。净错误减少不必提高F1，不以平均或.003微小差距重设判据。

{score_table(summary, ablations)}

对整窗partial，减少180/218FP、少110/111TP，净纠错70/107；对49邻域减少168/206FP、少110/111TP，净58/95。对同日数和去局部视觉两遍均提高，表明位置与局部回答有开发价值，却未稳定保住相对邻域的主F1。成功弃权保留在小试活动F1无可见增益；完整扩展相对strict每遍多54TP与20FP、净+34，F1 .7540对.7370、.7418对.7243。这是复用旧机制的实际收益，不能改称新模块首创。

事件IoU.3 .848485/.823529与邻域一致；IoU.5却.666667/.647059，低于邻域.787879/.705882。核心起止MAE14.07/20.86、22.71/11.64日，不能单独据均值声称定位全部改善。活动内阶段F1 .475411/.458085，低于邻域.492245/.475953；同阶段头差异来自活动覆盖，不是独立A/S/D模型升级。主解码前414/373次日期变化，权限外传播4/0日。

八族逐条结果见analysis-v1/family-summary.csv。单次形变第一遍FP283→87，快速形变127→30，复发116→56；混合跳变仍128/129FN，混合尖峰231FN。第一遍纯干扰FP零，但第二遍step_only仍21FP，不能宣称排除全部跳变。原独立干扰存在F1 .756098、偏移.400000、尖峰.877193继续单独报告；干扰输出不否决形变，未将干扰指标当RCRA活动收益。

主40/38请求、124449/116370报告token，对固定72/72及254540/254323，直接各24。权限变体使用相同回答，权限收缩本身不省调用。新增114请求412485token、0传输或解析失败、160上限；当前使用53原缓存请求。167使用请求payload/图片/提示、36方法864输出与评分精确重放，重放新增API零。9业务测试通过、14上游弃用warning，集中Ruff通过；这些检查不能证明参考独立或模型训练字节复现。

![统一日与事件比较](../artifacts/gnss-deformation-diversity-2026-10-12/daily-authority-report-v1/comparison.png)

原图事后复核0401见E持续变化后平台：撤回450—630附近大段误报，但630后仍残留，且相对直接撤回1个参考阳性日[450,451)。0399是相对整窗TP损失最大例：N上升/U下降同时E跳变，局部将[450,629]全判活动；权限拒绝51FP，也拒绝整窗恢复的45TP [564,609)。不能因数值冲突外弱变化就认定静止，亦不据模糊图修改生成参考。实际看图与半开错误追踪另存daily-authority-session-v1/actual-visual-review.md。

"""
    text = text.replace("## 6 讨论", result + "## 6 讨论", 1)
    discussion = (
        "### 6.7 修订权限的明确价值与尚未解决的损伤\n\n"
        "原日权限减少局部错误扩展，完整开发相对直接视觉有收益，但限制新增活动过窄，"
        "同时拒绝可恢复TP与严格事件定位。两阶段、scope、partial弃权保留和数值工具均已有近邻或仓库实现；"
        "当前只可称验证了具体权限取舍的候选机制，不能称稳定可发表贡献。"
        "下一待验证假设是新增活动与撤回活动是否应由不同证据授权；必须先探索已有单向修订和事件保护、"
        "证明独立差异并做同回答匹配控制，不直接增邻域或工具救分。\n\n"
        "完整24仍已暴露开发，每族3条；日期相关、两遍回答非独立数据。统计11项谬误检查另存，"
        "不从点估计宣称显著性、因果或现场泛化。失败不表示收益为零，但保留集不用于挑有利结果。"
        "投稿仍需明确技术差异、未参与选择数据上的稳定主收益或明确可靠性改善，"
        "并客观保留强数值.844167差距与IoU.5损伤。AI看图可完成开发分析，不能自称独立领域专家。\n\n"
    )
    text = text.replace("## 7 结论", discussion + "## 7 结论", 1)
    conclusion = (
        "\n最新RCRA小试正结果扩展后出现主F1取舍：相对直接视觉减少误报并提高日／IoU.3事件，"
        "但第一遍未胜邻域、少检110/111个整窗可恢复TP且IoU.5受损。按预设停止固定版，"
        "不更名较好控制、不改判据；当前贡献仍是可追溯候选和错误机制证据，独立确认尚未完成。\n\n"
    )
    text = text.replace("## 数据、代码与AI使用说明", conclusion + "## 数据、代码与AI使用说明", 1)
    reproduction = (
        "\nRCRA由probe/run/verify/report_landslide_daily_authority相关程序复现；"
        "daily-authority-probe/replay-v1保存小试64/512，expansion/replay/report-v1保存完整36/864及114新请求，"
        "全部局部回答在权限控制中共享。官方新检索／原文和实际阅读、看图、统计边界保存，"
        "受控参考不因预测重标。本稿所有公式使用LaTeX；渲染检查不证明公式语义。\n\n"
    )
    text = text.replace("## 参考文献", reproduction + "## 参考文献", 1)
    text += (
        "\n[19] Correia L, Goos J-C, Bäck T, Kononova A V. DQS: A Low-Budget Query Strategy for Enhancing "
        "Unsupervised Data-driven Anomaly Detection Approaches. [arXiv:2509.05663v4](https://arxiv.org/abs/2509.05663v4), "
        "2026修订。实际读官方摘要与第4节起始，未读全部算法／实验或运行；相关DOI元数据已定位，未另核出版社全文。\n"
    )
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v12.md"
    if target.exists():
        raise FileExistsError("Preserve complete manuscript versions")
    target.write_text(text, encoding="utf-8")
    snapshot = base / "manuscript-v12"
    snapshot.mkdir(exist_ok=False)
    (snapshot / "draft.md").write_bytes(target.read_bytes())
    bindings = read_json(base / "manuscript-v11/evidence-bindings.json")
    paths = (
        run / "prediction-freeze.json", run / "analysis-v1/freeze.json", replay / "freeze.json",
        report / "freeze.json", base / "daily-authority-probe-v1/freeze.json",
        base / "daily-authority-replay-v1/freeze.json", base / "revision-authority-literature-v1/freeze.json",
        base / "revision-authority-primary-v1/freeze.json", base / "daily-authority-session-v1/reading-notes.md",
        base / "daily-authority-session-v1/actual-visual-review.md",
        base / "daily-authority-session-v1/statistical-interpretation.md",
    )
    for path in paths:
        bindings[str(path.resolve())] = sha(path)
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError("Manuscript evidence binding changed: " + path)
    write_json(snapshot / "evidence-bindings.json", bindings)
    write_json(snapshot / "manifest.json", dict(
        source=str(target.resolve()), source_sha256=sha(target), decision=decision,
        contribution_validated=False, independent_field_reference=False, reserved_content_loaded=False,
        files={p.name: sha(p) for p in snapshot.iterdir()},
    ))
    print(dict(manuscript=str(target), evidence_bindings=len(bindings), decision=decision["status"]), flush=True)


if __name__ == "__main__":
    write(Path(__file__).resolve().parents[1])
