"""Bind the signed-event and target-fidelity results into a complete LaTeX-MD draft."""

from __future__ import annotations

import json
from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json


def write(root):
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    run = base / "signed-event-authority-probe-v1"
    replay = base / "signed-event-authority-replay-v1"
    report = base / "signed-event-authority-report-v2"
    fidelity = base / "visual-target-fidelity-v1"
    for folder in (run, replay, report, fidelity, base / "local-motion-literature-v1", base / "local-motion-primary-v1"):
        verify_manifest(folder, "freeze.json")
    if read_json(replay / "verification.json")["status"] != "VERIFIED":
        raise ValueError("Exact signed-authority replay required")
    session = base / "signed-event-session-v1"
    session.mkdir(exist_ok=False)
    (session / "passport.md").write_text(
        "# 方向分离事件授权与目标事实可读性\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent；用户授权自主研究。\n"
        "- Scope: 48训练／24暴露验证；两遍缓存局部回答；保留24输入／参考内容未加载。\n"
        "- Outcome: STOP_SIGNED_EVENT_AUTHORITY；STOP_LINEAR_TARGET_EVIDENCE。\n"
        "- Verification: SERA48方法1152输出与方向权限独立重构、统一评分精确重放；"
        "目标探针9628／14630查询、导出线性权重重算容差1e-10通过。\n"
        "- Reference: 过程参考；Codex视觉与数值复核是开发者分析，未修改参考。\n"
        "- 新API／新记录均0；两遍回答、日数与查询不是独立数据规模。\n"
        "- 研究目标仍active；两项固定负结果不进入扩展。无独立发表效果结论。\n",
        encoding="utf-8",
    )
    (session / "actual-visual-review.md").write_text(
        "# 实际原图复核与数值辅助\n\n"
        "按第一遍相对RCRA净纠错最小／最大规则，实际用view_image打开0395和0408四联诊断图。"
        "输出图包含过程参考，仅作开发者事后解释，不用于盲标或发送VLM；没有重标参考。\n\n"
        "0395：N/E约600日后出现持续增长、U下降，约660—680后趋于平台；此前较长时间围绕旧水平波动。"
        "SERA把候选[540,681)整体有序运动支持扩展为逐日新增。原块中心546.5—672.5，"
        "Kendall .955556、8正变化、最大正变化占比.248075，均能通过事件规则，"
        "却不能确定其中前段正在运动。数值追踪两遍相同：新增FP[540,564)及[565,595)，"
        "共54日；新增TP[595,597)共2日，净−52。此例直接支持事件证据过粗的局限，"
        "不据图调整门槛或边界。\n\n"
        "0408：N增长／U下降期间叠加尖峰，随后平台仍有尖峰；E整体变化较小。"
        "SERA与RCRA此例最终活动完全相同，净0；它是最大值例而非正收益例。"
        "不得将局部不同色带解释成新事件或正确修复。\n\n"
        "完整两遍：对RCRA分别新增105／123FP，补47／58TP、撤24／19TP，净−82／−84。"
        "28候选检查仅13个块数不足和15个阳性支持；没有触发其他拒绝原因。"
        "累计新增权限162／199日并不证明位置选择有效：主输出与居中控制逐字段一致。"
        "相对whole撤回范围不同，公共解码后也完全相同，须报告这种退化。\n",
        encoding="utf-8",
    )
    (session / "reading-notes.md").write_text(
        "# 本轮定向文献阅读边界\n\n"
        "三个arXiv官方搜索均200；关键词为time series visual anomaly detection、GNSS offset detection、"
        "time series anomaly localization。不是系统综述。GNSS搜索返回的大多是导航／攻击／对准，"
        "不能据此声称已覆盖长期位移设备偏移研究。\n\n"
        "MORA（2610.09473v1，2026-10-07）：实际读官方摘要、引言及3.1—3.2.3起始"
        "（保存文本约225—485行），确认两嵌套观察重建同一短目标、上下文提供解释性纠正。"
        "它将漂移当可解释正常演化；本任务持续坡体运动是目标形变异常，不能直接压低所有连续变化。"
        "没有读完cross-view probe、损失及全部实验，未运行作者模型。"
        "本页仅定位到arXiv通用GitHub链接，未定位作者代码，不断言不存在源码。\n\n"
        "TimeRLM（2608.03391v2，2026-09-06修订）：实际读官方摘要、"
        "3.1起始（保存文本463—500行）、实验设置860—885行；全文另定位作者链接"
        "https://github.com/OpenTSLM/TimeRLM。确认原序列可通过Python/视觉逐步查询，"
        "作者默认禁用子Agent递归；本项目无需多Agent或强化学习才能研究局部证据。"
        "作者报告的IoU不能与本项目过程参考日F1直接比较。\n\n"
        "作者源码索引首次GitHub API访问返回403 rate limit exceeded；原错误响应保存在"
        "timerlm-code-index-v1/repository.json。本轮未读取作者源码、未取得commit、未运行其模型；"
        "不自动重试，也不称源码复现。该可选访问限制不阻塞本地研究。\n\n"
        "仓库已有signed_block_facts、窄／宽块图、stationary/step/ramp竞争、"
        "PELT分段、phase15端点等停止方案，本轮实际查看相关源码与结果。"
        "不把这些旧操作重包装为新贡献。新增目标事实线性探针只是受MORA同目标原则启发的"
        "必要信息检查，不是MORA复现或新异常识别算法。\n",
        encoding="utf-8",
    )
    (session / "statistical-interpretation.md").write_text(
        "# 描述性统计与11项谬误覆盖\n\n"
        "1. 独立性：日／相邻查询相关，两遍不独立，未把1152输出当1152记录。\n"
        "2. 选择偏差：24验证已反复暴露，极值图是事后诊断，不代表总体。\n"
        "3. 多重比较：未报p值或挑最好变体，主方法和门槛在评分前固定。\n"
        "4. 因果越界：同回答权限控制隔离决策逻辑，不能证明现场因果。\n"
        "5. 显著性代替效果：只报计数、F1和MAE/R²，未称统计显著。\n"
        "6. 非显著等同无效：停止固定版本，不推断全部视觉方案无效。\n"
        "7. 聚合掩盖分组：保存八族与逐例表，纯干扰非损不能替代总体收益。\n"
        "8. 分母及基率：过程参考已知日／缺测规则不变，类别比例不是现场发生率。\n"
        "9. 缺测和弃权：未知不补确定活动，阶段保持活动门控与相同解码。\n"
        "10. 指标偷换：IoU.5一遍改善不能抵消主日F1下降；事实R²不是异常F1。\n"
        "11. 拟合与泛化：探针只TRAIN拟合，仍是已见开发验证；"
        "固定线性读出失败不能否定非线性可读性、局部像素或专用时序模型。\n",
        encoding="utf-8",
    )
    decision = read_json(run / "decision.json")
    summary = read_json(run / "summary.json")
    text = (root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v12.md").read_text(encoding="utf-8")
    text = text.replace(
        "# GNSS形变异常的有界视觉复核：逐日修订权限与机制验证",
        "# GNSS形变异常的有界视觉复核：局部证据与修订边界的机制研究", 1,
    )
    first_end = text.index("\n\n## Material Passport")
    text = text[:text.index("\n\n> ")] + (
        "\n\n> 中文完整研究初稿，2026-10-13，v13。整体形变异常为主，跳变／尖峰可同现且另识别。"
        "保留v12全文研究证据，新增SERA方向分离授权及目标事实可读性两项固定负结果；"
        "96×730、48／24／24不变，未打开保留内容。全部公式采用LaTeX，尚无独立稳定Agent/VLM贡献。"
    ) + text[first_end:]
    text = text.replace("Version Label: deformation-manuscript-20261013-v12", "Version Label: deformation-manuscript-20261013-v13", 1)
    text = text.replace(
        "当前证据支持减少错误扩展的开发价值及召回取舍，尚不宣称独立现场效果或保证发表。",
        "继而检验SERA方向分离事件授权，两遍日F1为0.7386／0.7289，新增误报多于恢复活动，"
        "相对RCRA净损82／84日。现有冻结视觉表示的局部事实线性重建平均R²为−0.0383，"
        "不能支持将此证据读出直接接入异常修订。两项固定方案停止。当前证据支持减少错误扩展的"
        "开发价值及召回取舍，尚不宣称独立现场效果或保证发表。", 1,
    )
    text = text.replace(
        "reserved records remain unopened.",
        "reserved records remain unopened. A subsequent signed event-authority test attains F1 of 0.7386/0.7289 "
        "and adds 82/84 errors relative to daily authority. A fixed linear probe of target changes from frozen "
        "visual embeddings attains mean R-squared of -0.0383. Both fixed mechanisms are stopped; the probe "
        "measures factual readability, not anomaly-detection performance.", 1,
    )
    method = r"""### 3.17 SERA：新增与撤回的方向分离事件授权

SERA（Signed Event Revision Authority）保留原日冲突集合 $C$ 作为撤回权限，新增则检验局部视觉阳性候选事件 $E$。事件证据 $H(E)$ 复用已有14日块有序累计、净位移／噪声、速率及非集中变化规则，不把该规则、单向修订或Agent框架称为新颖。设局部partial结果为 $\hat y^{(L)}_t$，候选新增种子与授权为：

$$
Z(E)=\mathbf1\!\left[\exists t\in E\cap C:
\hat y^{(0)}_t\neq1,\ \hat y^{(L)}_t=1\right],\qquad
A_t^+=\mathbf1[t\in E]\,H(E)\,Z(E)\,
\mathbf1[\hat y^{(0)}_t\neq1,\ \hat y^{(L)}_t=1].
$$

$$
\hat y'_t=
\begin{cases}
1,&A_t^+=1,\\
\hat y^{(L)}_t,&t\in C,\ \hat y^{(0)}_t=1,\ \hat y^{(L)}_t\neq1,\\
\hat y^{(L)}_t,&t\in C,\ \hat y^{(0)}_t\neq1,\ \hat y^{(L)}_t=-1,\\
\hat y^{(0)}_t,&\text{otherwise}.
\end{cases}
$$

所有权限仅限原观测日、选中目标；失败可保留未知，成功弃权使用原partial。公共持续期解码和48TRAIN阶段头不变，最终输出可能受解码传播影响。全窗撤回、取消事件支持、同新增日数居中、strict弃权及无局部VLM为匹配消融。后者仍含全局VLM，不称完整纯数值。原36方法保留，新增12流程共48方法1152条记录输出，不是1152条数据记录。

### 3.18 目标事实可读性探针：不是新增异常检测器

在继续接入视觉数值修订前，检验原512维冻结MobileCLIP2表示能否读取同一查询附近的有符号块变化。原13个14日块记为 $b_{t,j}\in\mathbb R^3$，三轴除原噪声代理；只保留第5—7块均至少8完整观测及原图实际编码的日期：

$$
d_t=[b_{t,6}-b_{t,5};\ b_{t,7}-b_{t,6}]\in\mathbb R^6,\qquad
\hat d_t=B\,\operatorname{Scale}_{\mathrm{TRAIN}}(v_t)+a.
$$

标准化和六输出Ridge只用48TRAIN查询拟合，固定 $\alpha=1$，无调参。零变化、TRAIN均值和固定种子打乱TRAIN图表示为对照。未读取活动／阶段／干扰标签或潜在形变，未编码新图或调用VLM。原视觉表示来自包含目标的观测图，因此这是事实读取而非未来预测；不能用其R²替代活动F1。必要继续条件为MAE均优于零与均值、优于打乱，且六维R²均大于0。通过也只允许设计新的独立冻结异常实验，不自动构成方法贡献。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    methods = ["query_motion_mass"]
    for rep in range(2):
        methods += [f"{name}_r{rep}" for name in (
            "visual_daily_partial", "sera_signed_event", "sera_whole_event",
            "sera_signed_no_event", "sera_signed_center", "sera_signed_strict", "sera_signed_no_local",
        )]
    result = (
        "### 5.19 SERA：候选整体运动支持不能定位内部活动日\n\n"
        + score_table(summary, methods)
        + "\n\n两遍日F1 .738574／.728938，低于RCRA .753963／.741811。"
        "相对RCRA去FP均0，新增FP105／123、补TP47／58、撤TP24／19，净−82／−84。"
        "主与全窗撤回／同新增日数居中控制最终活动、阶段及状态逐字段相同，"
        "无证据表明事件规则选择了更有价值的位置。主新增权限162／199日，"
        "候选检查15个有运动支持、13个块数不足；其余拒绝原因未出现。"
        "IoU.5 .7500／.7273，仅第二遍满足恢复条件；IoU.3 .8125／.7879反而低于原逐日权限。"
        "按评分前条件STOP_SIGNED_EVENT_AUTHORITY，不改变规则救分，不扩本版新数据。\n\n"
        "实际检查0395／0408图。0395候选[540,681)后半确有运动，使整体Kendall .955556、"
        "非集中变化等通过，但新增54个起变前阴性日，仅补2TP，净−52；"
        "0408尖峰可与形变同现，两方法此例净0，并非正收益。半开原日历变化范围另存，"
        "没有借助预测修改参考。原1152输出、方向权限与同日数控制及评分精确重放；"
        "旧167请求payload绑定继承上游已完成核验，0新API。"
        "主历史调用40／38、token124449／116370，本轮复用而非重新支出。"
        "全部48记录×两遍处理计时总1.384秒，含六变体与解码，不当主方法独立部署成本。"
        "九项相关业务测试通过，集中Ruff通过。\n\n"
        "### 5.20 视觉目标事实探针：停止固定线性读出\n\n"
        "| 方法 | MAE（噪声单位） | RMSE | 六输出平均R² |\n"
        "| --- | ---: | ---: | ---: |\n"
    )
    for name, row in read_json(fidelity / "summary.json").items():
        result += f"| {name} | {row['mae_noise_units']:.6f} | {row['rmse_noise_units']:.6f} | {row['r2_uniform']:.6f} |\n"
    result += (
        "\n9628TRAIN／14630开发验证查询、六原轴有符号变化；固定拟合25.564秒，"
        "导出线性权重重算1e-10通过。主MAE高于零、均值和打乱；只有两项U变化R²为正，"
        "四项N/E为负，STOP_LINEAR_TARGET_EVIDENCE。既没有异常F1提高也没有新的Agent结果，"
        "不接入主识别，不改alpha／seed／读出模型救分。失败只否定固定全局表示线性读取，"
        "不能推断非线性、局部像素、专用时序编码器或全部视觉方法不可用。\n\n"
    )
    text = text.replace("## 6 讨论", result + "## 6 讨论", 1)
    discussion = (
        "### 6.8 停止事件粗授权与固定线性证据的边界\n\n"
        "SERA说明“事件期间存在连续运动”与“期间每一日属于活动”不可等同。"
        "当前没有稳定的局部新增证据，不再排列权限或重包装旧块图／竞争曲线／PELT切片。"
        "最新MORA[20]通过同短目标的局部／上下文重建控制解释性纠正，"
        "但其正常漂移任务不能直接迁移为压低GNSS持续形变；TimeRLM[21]用代码／视觉查询"
        "原序列进行精确定位且默认不用子Agent递归，支持有限观察路线而非复杂Agent架构。"
        "本轮仅阅读规定段落，未复现其模型；TimeRLM源码访问403，不能称已读代码。\n\n"
        "目标事实探针给出了明确的下一步约束：在提出新融合之前，先证明其读取的目标证据"
        "可用并与已停止机制有实质差异。当前稳定方法、未参与选择数据的验证和独立领域参考仍缺。"
        "这些负结果不等于永远无法发表，但现稿不能写成已验证优于强方法的创新成果。\n\n"
    )
    text = text.replace("## 7 结论", discussion + "## 7 结论", 1)
    text = text.replace("## 数据、代码与AI使用说明", (
        "方向分离事件授权在同缓存完整开发中未稳定修复原权限的召回取舍；"
        "固定视觉目标事实读出也未通过信息条件。两项停止，保留全部结果，"
        "研究仍围绕整体形变异常，弱运动与A/S/D不回到主目标。\n\n"
        "## 数据、代码与AI使用说明"
    ), 1)
    text += (
        "\n[20] Mou X, Wang T, Wang R, et al. MORA: Modeling Observed Changes for Drift-Robust Time-Series "
        "Anomaly Detection. [arXiv:2610.09473v1](https://arxiv.org/abs/2610.09473v1), 2026-10-07。"
        "实际阅读摘要、引言、方法起始；未复现。\n\n"
        "[21] Zumarraga N, Steno L, Wang N, et al. TimeRLM: Recursive Language Models Enable Precise "
        "Anomaly Localization in Long-Context Time-Series. [arXiv:2608.03391v2](https://arxiv.org/abs/2608.03391v2), "
        "2026-09-06修订。[作者仓库](https://github.com/OpenTSLM/TimeRLM)已定位，API访问403，未读源码／运行。\n\n"
        "SERA复现入口：probe/verify/report_landslide_signed_event_authority.py；"
        "src/gnss_sim/landslide_signed_event_revision.py保存方向授权。"
        "probe_landslide_visual_target_fidelity.py保存固定事实探针、位置、权重及全部输出。"
        "signed-event-session-v1保存看图、统计与文献阅读边界。以上0新API／新数据；"
        "全部原冻结版保留，首次只读及报告接口错误披露于session收尾记录。\n"
    )
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v13.md"
    if target.exists():
        raise FileExistsError("Preserve all full manuscript versions")
    target.write_text(text, encoding="utf-8")
    snapshot = base / "manuscript-v13"
    snapshot.mkdir(exist_ok=False)
    (snapshot / "draft.md").write_bytes(target.read_bytes())
    bindings = read_json(base / "manuscript-v12/evidence-bindings.json")
    paths = (
        run / "freeze.json", replay / "freeze.json", report / "freeze.json", fidelity / "freeze.json",
        base / "local-motion-literature-v1/freeze.json", base / "local-motion-primary-v1/freeze.json",
        base / "timerlm-code-index-v1/repository.json", session / "actual-visual-review.md",
        session / "reading-notes.md", session / "statistical-interpretation.md",
    )
    for path in paths:
        bindings[str(path.resolve())] = sha(path)
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError("Prior manuscript evidence changed: " + path)
    write_json(snapshot / "evidence-bindings.json", bindings)
    write_json(snapshot / "manifest.json", dict(
        source=str(target.resolve()), source_sha256=sha(target), decision=decision,
        target_fidelity_decision=read_json(fidelity / "decision.json"),
        contribution_validated=False, independent_field_reference=False, reserved_content_loaded=False,
        files={path.name: sha(path) for path in snapshot.iterdir()},
    ))
    print(json.dumps(dict(manuscript=str(target), bindings=len(bindings)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    write(Path(__file__).resolve().parents[1])
