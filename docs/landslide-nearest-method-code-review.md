# TAMA与AnomaMind最近邻控制策略的官方代码核查

## 2026-10-09：方向复核停止与特征生成最近邻

[方向开发](landslide-direction-review-results.md)两版主组无识别增量且覆盖下降，全文v15；同信息方向XGB .7350的小幅日收益伴事件／边界退化，不改称Agent贡献。新官方arXiv检索8条，Crossref TLS EOF失败原样保存。已读[LLM-Generated Feature Pools](https://arxiv.org/html/2609.21801v1)摘要及3.1—3.5：调参窗口图生成特征，严格tuning选择，最终转导MAD检测；未定位作者库或复现。一般LLM特征生成已有先例，方向门控的失败也不构成新有效方法。读取范围与源哈希见literature-v1/direction-review-search-v1/reading-notes.md。


最新[阶段原始证据研究](landslide-stage-contrast-review-results.md)：开发修正105/46日、无正确损失，新12条冻结确认两遍0触发，未升级该选择器；完整稿v12。TSHarness与综述说明有限证据循环、工具分工、对基础检测器不劣化已有先例，实际阅读边界见新检索notes，不将通用机制当首创。

## 2026-10-09补充：MORA与TimeRLM

定向检索并读取MORA的双视图、式8—9单侧纠正及实验协议。上下文仅在改善同一目标重构时减去部分局部异常分数，受约束纠正已有先例；其Best RPA-F1／VUS不能直接与本文固定验证阈值的GNSS持续活动F1比较。未找到作者源码链接，不把arXiv反馈GitHub当作者实现。

TimeRLM全文文本给出作者库[OpenTSLM/TimeRLM](https://github.com/OpenTSLM/TimeRLM)，已核查commit `31fcd847b7cb37a1f1e6859b1dca7973ed0eae74`，取得README、注入变换、评估入口、绘图、视觉REPL和复现文档。实际源码`rlm.py:270`向模型提供信号文件并用numpy／scipy查询，`:410`的look把沙箱内图像回送；默认不启用子模型，根turn预算可配置。`anomalies.py:133`公开实现ramp后保持末端平台，所以这一生成形态也不是本文首创。泛异常level_shift在其任务为异常，本文单仪器阶跃则为非持续活動，语义不能直接混用。

这些是定位源码与相关段落的核查，未执行作者方法或复现发布指标。README明确缺模型权重、部分原训练补丁和PhysioNet预处理；不把取得源码写成完整复现。本文差异继续限定为GNSS两次原始持续位移冲突对有限观察位置及±49日可改写日期的联合约束。读取范围、首轮树SSL失败和成功版本、原字节及哈希见`literature-v1/background-validation-search-v1/reading-note.md`。

2026-10-09。下载并逐段检查官方仓库 `ChongKaKam/TAMA`，树SHA `db99a75a83d0ef05caf7ff0d199b75b594e98584`，原字节及哈希保存于 `artifacts/landslide-adaptive-review-2026-10-09/literature-v1/tama-code-v1/`。这是代码核查，不是完整效果复现。

## 已确认的实际流程

`main_cli.py` 第45–46行：正常参考默认3份，`--double_check` 默认关闭，需要显式启用。第110–137行从同数据通道训练图抽正常参考；如果训练路径不存在，则按test_structure中的normal条目抽测试图。这类回退不能直接用于本项目的独立评价，未来适配应只取共同训练分组的参考。

第471–487行先发正常模式提取请求，将normal_pattern答案保留在后续对话。第500–513行执行窗口异常检测；可选复核要求初判abnormal_index不为 `[]`。第544–582行的实际复核固定使用前2/3及后2/3重叠视窗，合在一个额外请求中，并提供先前检测回答。上方依据异常位置决定局部视窗的分支处于注释状态，不能写成该commit实际运行了预测中心自适应选窗。

泛异常提示分类包括global、contextual、frequency、trend、shapelet，输出带1–4置信等级的字符串区间。它不是本项目“原始持续位移活动＋未知”的同任务语义。源提示称纵轴为索引、横轴为数值；实际主路径rotation_angle=0、ImageConvertor通常按常规x索引绘图，适配时必须明确坐标语义，不静默修正后再称原样复现。

`Datasets/Dataset.py` 第819–854行存在用真值扩充命中事件的point adjustment，评价函数默认关闭；不能把代码中存在该函数写成所有官方结果必然使用它。本项目共同评分禁止这种标签依赖调整。

请求控制有失败重试、某些模型使用额外JSON格式化模型。当前项目无重试且严格保留失败。原库ChatBotMAP也没有当前qwen3.8-flash入口；模型替换、参数、正常参考、重试和任务接口都需要明确记录。不能把当前直接视觉控制称完整TAMA。

## 与当前方法的具体差异

当前GNSS方法用原始持续运动冲突一次性选择观察位置，初判为空时也可因原始持续变化而复核潜在漏检；TAMA该commit的可选复核由初始非空异常触发。后者主要修正已有候选，当前策略尝试在有限预算下同时发现初判漏检区域。TAMA固定重叠2/3多尺度观察与当前冲突排序的最多两个不相交180日目标也不同。

这些差异有可定位源码依据，比“使用Agent”“加局部图”“调提示”更具体。但AnomaMind和其他工具Agent也可能主动取证漏检；不能从TAMA一条代码路径推出首创声明。已有[强基线与消融](landslide-adaptive-review-results.md)验证了普通视觉控制下的选择价值，仍需最近邻同任务适配比较来评价创新强度。

## 可执行比较边界

合理的下一项比较是TAMA正常参考＋初判阳性触发多尺度复核的**GNSS任务适配**：正常参考仅共同train；同原始数据和VLM；统一活动／未知语义、时间轴及学习阶段头；同时保存无复核与有复核输出，全部调用计入成本。需要公开从原算法变更的三分量画法、活动定义、严格JSON、模型与重试政策。此适配的成绩不能填写为原论文已发布完整方法成绩。

后续已实际运行该GNSS适配，见[另12记录统一结果](landslide-interval-consistency-results.md)和`tama-gnss-confirmation-v1`。共同train的0001／0007／0013作为正常参考，每遍正常描述一次按同分布共享；初判、阳性触发首尾2/3复核分别保留。对话按前文文本及保留图片展开，采用共同Qwen／参数／无重试／阶段头，原任务分类与绘图已明确适配。44请求／449660 token，无传输／解析失败，正常模式调用已计入；原通用异常基准和发布模型仍未复现。

新确认适配有复核F1 .7732／.7748，原默认 .7852／.8365，但事件、FP和边界没有全面优势，不能据此推广全面优胜。原默认强后弱case0072各恢复162弱日、适配各0，而弱恒速case0063适配208、默认146／82，场景差异必须完整报告。严格整段检查在缓存后新确认失败，未升级默认；现场无参考仍不算准确率。

## AnomaMind当前官方源码：数值复核已有先例

后续实际补读同commit完整tool.py／Evaluator.py，见`literature-v1/anomamind-rival-tools-v1/reading-note.md`。工具工厂实际注册stats／grep／value_zscore／diff_zscore，没有三运动描述叠图；Evaluator压缩取证后仅对严重误判／漏检／矛盾建议redo。此前“未阅读Evaluator”是当时边界，现已补齐；没有执行原模型，不能据一个工具表主张全领域首创。新竞争描述目标试验已因误报增加停止，见[结果](landslide-motion-competition-results.md)。

2026-10-09下载并实际阅读 `ustc-time-series/AnomaMind` 的main树 `bf2c0cb3aaeb5a47aa7acabd1714ce123ddcdaa1`，原字节、获取信息与哈希在 `literature-v1/anomamind-code-v1/`。核查覆盖Locator、Actor、Detector、AnomaAgent、finalize、WorkflowState与README；没有执行原方法，也没有复现其已训练检测器。

`Locator.py:157`提示纵轴为预处理后的缩放／去趋势值，判断突然形态而非慢漂移；`:205`读取单个segment预览，将粗视觉区间写入state。该任务与原始物理位移的缓慢持续活动语义有差异。不能把该提示直接运行后的分数当GNSS长期活动的公平完整基线，也不能由一条提示断言所有AnomaMind版本均不处理趋势。

`Actor.py:57–81`定义首次未主动发工具调用时的默认取证：stats、value_zscore(3.5)、diff_zscore(5.0)，有视觉候选时还取候选最小／最大端点外扩10索引的grep。`:168–215`的实际Actor流程最多8轮；**并非每次固定先调用这四种工具**，它们只在首轮模型没有工具调用时补入。Actor上下文要求用zscore／grep确认、拒绝或细化视觉定位。视觉候选为空也仍存在统计取证，所以本文不能声称所有已有Agent只会修复初判阳性。

`Detector.py:311–377`将Actor取证对话重放为文本，并输出最终异常答案；不是将原始局部图直接再次发给Detector。`AnomaAgent.py:83–117`明确串联Locator→Actor→Detector，默认启用Evaluator及返回Actor的有限重做，重做清空Actor／Detector对话，保留最初Locator提示。代码的迭代停止用 `iteration_count > max_eval_redo_iterations`，这里只记录实际比较，不把默认3静默写成精确最多3次重做。未下载／阅读Evaluator内部，故不推断其所有评价准则。

`finalize.py:146–184`解析最终Detector区间并导出，不见本文形式的逐日“冲突支持集合＋观测半覆盖→可修订掩码→范围外保留全局预测”合并。此结论仅覆盖所读文件及commit，不是全领域不存在相同设计的证明。README训练说明包含跨域SFT、RL和已发布0.6B／8B检测器；临时把Detector换成共同Qwen API会改变混合推理机制，不能称完整原方法复现或填入其发布成绩。

因此，创新表述收窄为**原始持续运动冲突对观察位置与局部修订权限的联合约束**。数值取证、粗到细定位、有限循环和工具框架本身均已有近邻。新候选保持原始图与VLM回答，只允许冲突日及±49日原始观测半覆盖内、且位于选定目标的日期被局部回答改写；数值不直接赋活动标签。等可修订有观测日数的居中对照检验冲突位置是否有增益。应由新冻结数据支持这一机制，而不是依据最近邻架构差异宣称首创。


## 2026-10-09新增：TimeEvo／LEARN-TS／CAPS

TimeEvo[14]把错误归类为测量缺口，生成仅提供证据的工具，并在适用问题范围复核原答；其准入同时统计修好和改坏。本文不将有限保护式修订、配对损益检查或失败驱动工具视为新颖。LEARN-TS[15]使用固定正常性语义参照辅助掩码重建，CAPS[16]从目标参照构造配对监督，说明正常参照及残差构造也已有相关研究。本候选使用训练正常条件的数值单谐波作附加图，仍须由GNSS全任务技术差异与实测收益证明贡献，不能仅以参照形式不同成立。

原文、作者commit／源码、实际阅读范围与失败原字节见`literature-v1/reference-observation-search-v1/reading-notes.md`。TimeEvo为QA问题级受保护复核和配对准入，不能把这些本身称新贡献；所读源码未见本文GNSS逐日冲突49日权限，不据局部阅读证明首创。

## 2026-10-09：ARTIST与历史尺度压缩准入核查

本轮定位[ARTIST作者库](https://github.com/mims-harvard/ARTIST)，commit2bbc28a86f346efb7311ddc227f8952c3aa8b21e；实际阅读论文引言/3.1—3.3、README部分、完整segment_memory.py与tools_qwents.py、推理入口前117行，未运行训练模型。有限主动选段、缩放与controller-reasoner已有公开先例，不作为本项目首创。金融多尺度视觉基准另提供跨尺度与XGBoost对照先例，其任务不支持GNSS四倍门槛。原字节/URL/hash及[阅读范围](../artifacts/landslide-adaptive-review-2026-10-09/literature-v1/scale-review-search-v1/reading-notes.md)保存。尺度准入[开发结果](landslide-scale-review-results.md)未通过全部登记条件，同预算位置对照更高，故不主张已建立创新或窗口选择优势。


## 2026-10-09：ARTIST奖励实现与实际纠错收益校准

同commit补读controller_reasoner_rewards.py前188行：consistency_score奖励、轮次惩罚、片段探索及格式奖励已实现，按下游效果学习片段选择已有先例；配置TLS EOF未读，未执行作者模型。[读取范围](../artifacts/landslide-adaptive-review-2026-10-09/literature-v1/review-utility-code-reading-v1/reading-notes.md)与原字节保留。新的[纠错收益开发](landslide-review-utility-results.md)小规模选窗通过但完整迁移失败；提案接受两遍与简单数值投票完全相同，不能以收益学习或数值投票包装Agent贡献。


## 2026-10-10：TSHarness及FreshCast补读

TSHarness官方PDF3.1—3.5已实读，源冻结的工具选择、答案收益－调用代价及有预算重新感知已有先例。FreshCast3.5／3.6及4节相关命题把修正与原错误对齐来校准权重，非收益保留原预测；其MSE／同分布独立假设不能移用为本项目可靠性保证。[原文、失败获取与读取范围](../artifacts/landslide-adaptive-review-2026-10-09/literature-v1/grouped-proposal-search-v1/reading-notes.md)保存，未定位作者源码、未复现。四背景校准也未建立稳定超简单数值投票的增益，不把学习框架本身当贡献。


## 2026-10-10：真实正常实例检索最近邻及否定实验

TAMA官方已存commit db99a75a83d0ef05caf7ff0d199b75b594e98584的main_cli.py 110—137、471—513再次实读：随机三正常图先提正常模式，非本30日增量相似检索。Anlu官方2610.06180v1实读摘要/3.1—3.3/4.1；MORA 2610.09473v1实读摘要/引言相关/3.1/3.2.1—3.2.5，3.2.6仅开头。两篇作者源码未能确认，不填作者模型成绩或无损保证。定向12条检索记录存在重复，不作系统综述纳入数量。实际正常参照检索两遍均失败，数值初判补充单遍提前停止，见landslide-normal-anchor-results.md；正常参照、ICL及单向修正已有先例，检索函数不当创新。原PDF/提取文字/读取范围保留于literature-v1/normal-contrast-search-v1。


## 2026-10-10：同记录最近邻适配补齐

官方最新TAMA/AnomaMindcommit与已读版本一致。原GNSS适配同源码在0121—0132追加44请求两遍，已统一41配置；复核F1 .7741/.7708低于主但FP更少、覆盖更高，主净确定纠错−200/+359。主调用更多、报告token更少，不能称整体可靠性或能耗优势。具体适配、追加时间、版本修复及[完整结果](landslide-tama-budget-comparison-results.md)保留，未复现作者发布模型。
