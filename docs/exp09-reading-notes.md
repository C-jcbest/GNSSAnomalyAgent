# EXP09：重复判断与返回对一致性调研

2026-10-14。动机来自EXP08已核实的服务响应变化，不预设多数一定更正确。

| 来源和读取范围 | 实际支持的思路 | 不可推出的结论 |
|---|---|---|
| *Can Multimodal LLMs Perform Time Series Anomaly Detection?* / AnomLLM，ICLR2025；已缓存方法及结论，本轮重新阅读结论 | 作者讨论模型特定检测能力与可能的集成方法；支持检验集成，轻量结构仍优先 | 未提供本轮同一视觉服务三次投票在GNSS有效的证据；其CoT实验也不支持默认加长推理 |
| *See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers* / TAMA，[arXiv:2411.02465v1](https://arxiv.org/html/2411.02465v1)；既有全文，本轮重读方法自反思与后处理段 | 多尺度自反思以稳定分析，重叠窗按类别投票，表明稳定性和多视角聚合值得研究 | 它的窗口置信度阈值与本轮候选2/3多数不同；没有运行其完整作者代码，不能称TAMA复现 |
| Wang等，*Self-Consistency Improves Chain of Thought Reasoning in Language Models*，[arXiv:2203.11171](https://arxiv.org/abs/2203.11171)；本轮新取得摘要 | 原方法采样多条推理路径并选取一致答案，提供聚合原则 | 作者任务是算术/常识推理；本轮不改温度、不启用CoT，属于受启发的重复视觉决策，不借用作者提升比例 |
| VLM4TS，AAAI2026；已核正式元数据及方法全文 | 候选定位与视觉复核分工是近期主要方法背景，本轮只修改复核结果聚合 | 本轮未运行完整ViT定位器，不填作者原方法同GNSS任务的成绩 |

Self-Consistency的OpenReview地址本轮HTTP200，但正文是“Verifying your browser”挑战页，
不是论文详情。因此不把该响应写成已读正式会议页；本轮直接可读证据是arXiv摘要。
新网页原字节和SHA保存在`papers/exp09-consensus/manifest.json`，
近期原文继续使用既有`papers/optimization-20261013/`、`papers/evidence/`缓存。
计划的多数修复是项目方法，分数需以新开发/正式站实测，不能把相关论文原任务成绩写入本项目表。
三次请求分别执行，但同一模型和同一输入可能有相关错误；不把请求分开称为统计独立，
也不把多数票解释为校准概率。如果成员完全一致，多数将没有收益而增加请求成本。
