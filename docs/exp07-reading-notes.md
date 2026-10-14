# EXP07：误差定向的论文调研

2026-10-14；问题是漏尖峰诱发假分段，以及阶跃/暂态共存时互斥后处理损失召回。
本轮为定向阅读，不是系统综述。来源原文、Crossref响应及SHA256见
`papers/exp07-robust-morphology/manifest.json`；既有近期全文缓存未覆盖。

| 文献与实际阅读 | 与实验相关的依据 | 使用边界 |
|---|---|---|
| Fearnhead、Rigaill，*Changepoint Detection in the Presence of Outliers*，JASA 114(525):169–183，2019；在线2018。[DOI](https://doi.org/10.1080/01621459.2017.1385466)，本轮获取并读[arXiv v2全文](https://arxiv.org/html/1609.07363)摘要、§1、§2.1、§2.2、§3和作者源码入口 | 平方损失可能把孤立异常拟合为额外变点；有界损失对极端离群点的鲁棒性与最短可检出片段有关。不能只把短片段一律删掉，否则会漏短真实事件。 | 作者[R-FPOP仓库](https://github.com/guillemr/robust-fpop)元信息/README已获取，README仅标题。本轮未运行其R/C++实现；ISC是本项目不同模型比较，不能称R-FPOP复现，也不能声称有界损失理论保证。 |
| He等，*Harnessing Vision-Language Models for Time Series Anomaly Detection*（VLM4TS），AAAI2026。[DOI](https://doi.org/10.1609/aaai.v40i26.39319)；复读既有arXiv v2方法（EXP06已有阅读）和本轮正式摘要 | 候选定位与全局视觉核验分工具有近期依据，数值候选漏检会限制后续视觉能力。 | 作者以预训练视觉编码器定位；本项目以训练RF/位移拟合定位，是不同适配。原论文F1-max与本项目冻结类型日F1不同，不横向搬用24.6%等论文数字。 |
| Zhou、Yu，*Can LLMs Understand Time Series Anomalies?*（代码AnomLLM），ICLR2025；本轮复读既有arXiv `2410.05440v3`摘要、引言与限制推理段 | 其受控实验中图像输入比数值文本更有用，显式解释/CoT并未稳定提升。 | 支持保持单轮简洁角色，测试“非互斥形态输出”而非增加复杂推理链。这不是说所有模型和任务CoT都无用。作者数据/模型与本轮GNSS不同。 |
| Wang等，*ChangePointCNN-GNSS: An AI Model for Assessing Change Points and Optimizing Site Velocity Estimation From Global GNSS Data*，JGR: Machine Learning and Computation，2025。[DOI](https://doi.org/10.1029/2025JH000910)；本轮复读正式全文§1–2 | GNSS解析候选加图像学习评估有明确领域先例，区分瞬时与过渡变点具有应用意义。 | 原研究会搜索解析阈值配置，并选至少4年无变点段估速；当前730日滑坡类型识别不采用其阈值循环、不声称同任务复现。 |
| *Outlier Detection of GNSS Deformation Time Series Based on VMD and MAD*，ICGMRS2026。[DOI](https://doi.org/10.1109/ICGMRS70230.2026.11542245)；本轮新检索并核实Crossref题名、会议与2026-04-17日期 | 是进一步关注GNSS分解加稳健统计的近期文献线索。 | **只取得元数据，没有实读全文、算法、阈值或源码**。不宣称它有效或无阈值，不填对比成绩，不用标题推断具体方法配置。 |

## 选择与取消逻辑

优先测试可由既有三轴观测、历史尺度与分段源码直接支撑的两项扩展。
不新增复杂噪声、模型微调、递归代理或跨站物理告警阈值。
ISC的`3p log(3n_local)`与`2 log(N_eligible)`是披露的统计惩罚，
不是作者默认，不表述为无参数。ISC-v1负结果表明“增加数值检测”并不自动改善。
非互斥输出允许模型同时识别永久/临时分量，但不能仅因数值两个边缘存在而保留永久阶跃，
须以实际开发对照判断是否减少漏检以及是否增加误报。

若结果不足以达到预先登记的0.02收益门槛，则回到已备份的EXP06生产实现。
后续可考虑真正有界损失的联合分段，但需先解决参数来源和三轴/活动适配，
不能把本轮失败改写成该未执行方向有效。
