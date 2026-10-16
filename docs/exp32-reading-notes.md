# EXP32文献复核与方法身份

2026-10-16。本轮是EXP31探索假设的独立开发确认，不是新提出通用视觉智能体架构。
仍采用academic-research-suite实验工作流；实验结果按实际执行记录，论文事实按指定来源区分。

## 本轮实际读取

1. **Can Multimodal LLMs Perform Time Series Anomaly Detection?**，arXiv:2502.17812v2。
   本次重新取回全文HTML（HTTP200），读取§5输入模态对比、§6开头及§6.1动机，
   实际SHA与URL见`papers/exp32-context-confirmation/retrieval.json`。
   原文报告文字侧重数值变化、图像侧重时序结构，并观察到其范围任务视觉优于文字/混合输入。
   这是原论文模型与数据上的观察，不是qwen3-vl-plus在GNSS的已知结论。
   本项目因此以同粗筛图文终判对照检验模态作用，同时保留去数值证据，避免把数值贡献算成图像贡献。
   §6提出scanning/planning/detection/checking框架，不能把视觉先行或多轮处理本身当本项目独有创新。
   本轮没有执行作者完整框架，也没有把其Table分数列为GNSS对照。
   来源：https://arxiv.org/html/2502.17812v2 。

2. **LLM Agents for Time-Series: A Survey**，arXiv:2608.26226v1。
   复读EXP31原始全文§4.3，重点是证据质量、对基检测器不退化及检测/诊断区别。
   本轮将模型识别机制与融合系统收益分列，不用自然语言解释或过程粗筛命中率代替识别成绩。
   来源：https://arxiv.org/html/2608.26226v1 。

3. **TimeEvo: Failure-Driven Self-Evolution of a Time Series Agent**，arXiv:2609.27277v1。
   复读EXP31原始全文的paired admission相关段落：同时统计helped和harmed，而非只看平均改善。
   本轮采用逐站新增TP、损失TP、误报及新数据确认；没有实现作者工具合成或QA准入统计量。
   来源：https://arxiv.org/html/2609.27277v1 。

## 检索与适配边界

本轮定向复核上述论文，没有声称再次穷尽最新文献；最新工作发现范围沿用EXP31的有限检索说明。
全局/合并局部图、35日上下文、最多两区域、confirmed/uncertain与数值软证据都是本项目适配，
不是宣称作者默认流程。EXP32不调整EXP31的提示、分辨率、坐标粒度和后处理，
以未预测站确认其联合上下文假设；稳定数值实现、模型、数据机制均不改。

形变没有新增视觉预测，不能借形变Aff分数声称视觉已能识别持续活动或内部加减速阶段。
只有新开发指标通过预登记条件，才继续扩大机制确认；完整系统是否升级另行判断。
