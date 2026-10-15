# EXP26实读与设计依据

读取日期2026-10-15；定向复核三篇已识别近邻，不是系统综述或穷尽新论文检索。
本轮全部公开全文HTTP200；原HTML、抽取文本、URL、SHA保存在papers/exp26-counterfactual/retrieval.json。

| 论文、版本 | 实读位置和事实 | 本轮借鉴与边界 |
|---|---|---|
| *See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers*（TAMA），[arXiv2411.02465v1](https://arxiv.org/html/2411.02465v1) | Methodology / Multi-scaled Self-reflection：把先前异常输出与放大的异常局部图交给模型复核；作者强调新图可避免重复同一视图。Results Aggregation还使用置信累计阈值，表中包含point-adjusted F1。 | 本轮提供不同诊断视图而非单纯增加DPI；不照搬作者置信阈值或point adjustment，不把作者分数与本项目Aff直接比较。 |
| He、Alnegheimish、Reimherr，*Harnessing Vision-Language Models for Time Series Anomaly Detection*（VLM4TS），[arXiv2506.06836v1](https://arxiv.org/html/2506.06836v1)，2025-06-07 | §3.2 VLM4TS: Vision-Language Model-based Anomaly Verification and Refinement：ViT4TS给局部候选，VLM接完整序列图核验全局一致性，返回范围/置信并弃低置信。 | 候选定位与视觉解释分工有先例，不称本项目首次提出两阶段。原方法不是本轮的数值PELT，也不是候选扣除图；本轮未运行完整原方法。 |
| Xu等，*Can Multimodal LLMs Perform Time Series Anomaly Detection?*，[arXiv2502.17812v2](https://arxiv.org/html/2502.17812v2) | 摘要/主要分析指出MLLM对粗粒度异常与传统细粒度检测互补；不规则序列部分用缺失留白保留形状，讨论避免插值失真。范围主指标沿用此前已核实的Affiliation协议。 | 数值负责边界、视觉负责跨轴范围语义；缺测不填值。不能据论文的平均优势推断GNSS或当前模型必胜。正式出版年份/README冲突沿用EXP25披露。 |

## 本轮提出的可检验推论

原SDR有时把暂态内部永久阶跃当返回点；仅展示多个原图时，模型需要同时记住方向、幅值和背景运动。
因此试验“只按起点估计的offset扣除候选范围”：若候选正确，其终点应更加连续，同时内部独立阶跃可以保留。
起点以外的结束边界并未参与offset估计，避免通过拟合两端制造必然连续的图。
该反事实诊断是本项目设计，不是三篇论文的已证算法，也不声称已检索证明全球首次。

这仍含数值加工，不能把全部贡献说成视觉。控制包括相同逐点文本、原图、无图、
同候选数值近邻、原SDR，比较输入有无和最终范围改变，而不把模型生成理由视为实证。
特别保留Affiliation对邻近/部分覆盖区间的高分，以及日F1和FP代价，避免选择性解释。
