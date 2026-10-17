# EXP38调研与适配说明

2026-10-17，继续采用academic-research-suite实验流程。用户已授权设计实验、代码修改及
现有视觉接口执行，故不重复请求内部流程默认确认。材料不是系统综述，也不证明领域SOTA。

## 实际阅读的方法依据

1. **See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly
   Analyzers（TAMA）**，
   https://arxiv.org/html/2411.02465v1 。本轮重新取得全文，阅读§3.2与附录正常参考提示。
   其正常参考学习先让模型提取正常图像模式，再用于后续检测/分类，并有多尺度自反思。
   本项目一步给同站正常前缀图与目标图，没有学习正常描述的独立调用，也没有该论文完整
   后处理；属于借鉴，不能称TAMA源码复现。正常参考本身不是本项目首次创新。
2. **Can Multimodal LLMs Perform Time Series Anomaly Detection?**，
   https://arxiv.org/html/2502.17812v2 。本轮重读TSAD-Agents实验讨论和消融段：其混合架构
   结合传统数值方法的点识别和多模态模型的范围识别，包含Scanning/Planning/Checking等
   智能体与Chain-of-Scanning消融。本项目仅采用类型分工与图文消融的研究动机，未运行该
   作者完整智能体，不把其原数据结果当作GNSS可比成绩，也不把本项目单轮流程称完整复现。

两篇全文及纯文本新副本、HTTP状态和SHA保存于`papers/exp38-reference-residual/`。
抓取成功后打印片段时Windows默认编码发生UnicodeEncodeError；HTTP200全文及文本已经
保存成功，retrieval同时记录该本地输出错误，后续使用UTF-8实际读取，不虚报抓取失败重试。

## 本轮有界近期检索

arXiv API查询time series AND anomaly AND (vision OR multimodal)，按提交日期倒序前8条。
原查询及响应见search.raw/retrieval.json，返回133条总命中，但本轮仅筛选前8条。
其中CAPS（2609.33610v1）的摘要涉及context-anchored pairing；尚未阅读正文、核验作者
实现或执行，只作为后续线索。
MACTS-EM属于forecasting，其他跨模态研究未必以时序曲线图作为视觉输入，不能据关键词
相似直接作本项目近期VLM对照。本轮并不保证覆盖所有最新研究。

另外取得**LEARN-TS: LLM-Enhanced Alignment and Reconstruction with Normality Guidance for
Multivariate Time-Series Anomaly Detection**（2609.38789v1），
https://arxiv.org/html/2609.38789v1 。实际读取§3.2、normality-gated anomaly score和§4.3的
normality-reference control：其冻结语言模型产生观察语义与窗口无关的正常语义，
后者参与正常对齐和重建分数调制。正常语义对随机参考的额外收益被作者描述为modest，
不能据此假定本项目正常图参考一定改善。该方法不是直接读取GNSS曲线的视觉LLM；
本轮没有复现训练、校准、阈值或作者代码，不能填对比分数或称本方法超越它。
此正文在EXP38请求已经冻结后补读，不改变本轮提示/融合，只帮助解释后续取舍。

## 本项目新增适配

NRV复用已有正常前缀拟合与MAD，加入原始位移/残差双视图、显式可见正常参考及两个范围
类别，按分量/表示共享参考与目标纵轴。去参考消融仍保留MAD和拟合，不能说完全无历史。
点事件所有者固定；视觉范围只补不重叠的数值漏段，两范围可共存，不由视觉删除点事件。
噪声/生成器、数值拟合与作者决策参数不变。此为本项目提示、表示和融合适配，无新增跨站
毫米或速率阈值。收益需由冻结新数据验证，不能由论文动机直接推出。
