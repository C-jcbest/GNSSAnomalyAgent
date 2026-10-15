# EXP29：视觉先行流程的文献依据

检索日期：2026-10-15。实际检索了 arXiv API、OpenAlex，读取下列 arXiv 指定版本 HTML；不是系统综述，也不据此断言覆盖全部最新工作。原文和取回回执见 `papers/exp29-visual-first/`。下列均按预印本描述，不擅自赋予会议录用状态。

| 工作与实际读取版本 | 实际读取内容 | 可借鉴内容及边界 |
|---|---|---|
| [AnomaMind](https://arxiv.org/html/2602.13807v1) | §4，尤其视觉 interval localization、feature extraction 和 coarse-to-fine workflow；作者 Locator.py、Actor.py | 视觉归一化曲线先粗定位，随后工具取证与细检测。原框架包括强化学习决策，不等同本项目的免训练受控流程。 |
| [TimeRLM](https://arxiv.org/html/2608.03391v2) | 方法、精确定位与多轮交互分析；作者 README | 原信号留在代码环境，多轮数值查询与 look 绘图交替。包含后训练及不同变体；原文使用 IoU，不能宣称范围检测均不用 IoU。 |
| [DAST](https://arxiv.org/html/2606.06261v1) | 方法：VLM 曲线 profile →领域 LLM 筛查→高分辨率 VLM 热图区间 | 多阶段、不同分辨率证据有依据；O-RAN 场景及其评分筛选不能直接当作 GNSS 方法或结果。 |
| [SAGE](https://arxiv.org/html/2605.05725v2) | 框架及四类 analyzer、Detector、Supervisor | 类型专门工具及数值与诊断图融合；无需为模仿结构增加多个代理。 |
| [TimeEvo](https://arxiv.org/html/2609.27277v1) | 摘要、failure-driven evolution 与 paired admission | 更多工具、泛化反思未必改善；逐站核对新增正确与破坏正确的结果。原任务为时间序列问答，不能直接转用成绩。 |
| [Can Multimodal LLMs Perform Time Series Anomaly Detection?](https://arxiv.org/html/2502.17812v2) | §6 TSAD-Agents：scanning、planning、detection、checking | 扫描类型后选工具与模态，检查再细化。视觉先行、类型路由和 agent 本身已有先例，不应作为本项目唯一创新。 |

作者仓库核验：AnomaMind `bf2c0cb3aaeb5a47aa7acabd1714ce123ddcdaa1`；TimeRLM `31fcd847b7cb37a1f1e6859b1dca7973ed0eae74`。读取范围及 SHA 见 source-retrieval.json；仅下载阅读，不代表执行复现。最初 raw.githubusercontent.com 连接失败，改从 GitHub contents API 成功读取，未使用虚构内容补全。

本次要检验的研究问题：在跨站尺度不同、持续真实位移与暂态质量事件可叠加的 GNSS 场景，视觉自主提出范围，再由站内信息准则验证和细化，能否补回“仅在数值未配对阶跃之间复核”所遗漏的范围？视觉必须能产生数值自由候选之外的日期；模态贡献用同信息数值文字对照，流程贡献用数值先行视觉对照。

本次提出的是文献启发的 GNSS 适配方法，不是上述论文作者原方法复现。通用数值基线及原项目适配分别列出；没有执行的论文方法不填写分数。
