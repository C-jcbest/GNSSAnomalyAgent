# EXP20—22：区间评价与视觉定位实读笔记

2026-10-15，延续ARS academic-research-suite实验工作流。区分已确认文献事实、GNSS适配和本轮实验结果；模型生成的解释不当成事实证据。

| 文献/代码 | 本轮实际读取 | 可采用的思想与边界 |
|---|---|---|
| Huet, Alexis; Navarro, Jose Manuel; Rossi, Dario. *Local Evaluation of Time Series Anomaly Detection Algorithms*. KDD 2022. DOI:10.1145/3534678.3539339，arXiv:2206.13167v1 | 新下载全文，题名/作者/会议/DOI及方法、实现段；作者README、generics.py、metrics.py、区间积分与空集处理；完整原测试60项通过 | 使用局部距离概率的Affiliation P/R评价区间，代码半开区间；不需要位移阈值，也不需要为每个算法扫分数阈值。但其P/R不是区间IoU，近邻误报及碎片有时分数仍高；必须另看误报、IoU和覆盖 |
| Zhou、Yu，AnomLLM，ICLR 2025；本地全文`papers/optimization-20261013/anomllm_2025_text.txt` | 重读§4.2数值/图像表示、§4.3评价及图2（本地247–248行）、§5结论 | 作者明确LLM输出离散区间，使用Huet等指标的P/R调和均值为affinity F1；连续排名指标VUS不直接适用。本项目报告Affiliation，F1公式与跨站汇总写清楚，不能借论文排名证明GNSS效果 |
| *Can Multimodal LLMs Perform Time Series Anomaly Detection?*，VisualTimeAnomaly，arXiv:2502.17812v1；正式ACM Web Conference 2026记录DOI:10.1145/3774904.3792376 | 重读已缓存方法中的点型/区间型任务区别及作者提示接口 | 用户所指论文是VisualTimeAnomaly，不能与AnomLLM混同。本轮按区间任务分列暂态/形变；未完整运行其原数据/原提示全套流程，不把本项目结果标为作者默认复现。作者README年份冲突仍见EXP10笔记 |
| TAMA，*See it, Think it, Sorted*，arXiv:2411.02465v1 | 重读Multimodal Reference Learning、Multi-scaled Self-reflection、表1、Appendix A.3；本地`papers/evidence/tama_arxiv_text.txt`约489–499、4380–4440行 | 放大首次识别区间可提供新图像信息；EXP21据此将首次预测边界附近放大。其point-adjusted F1和置信聚合不移植；本项目没有因此设置位移阈值，也不宣称完整TAMA复现 |

作者Affiliation源码原样保留在`vendor/affiliation-metrics-py`，MIT许可，提交
`8d8449858096bbade6a6e70848d05c9cc9b846fe`。原zip、commit元数据、逐文件SHA和论文原HTML在
`papers/exp20-range-evaluation/`。最初尝试的无`-py`仓库、IntelLabs/TSAD_eval地址返回404，不能把它们写成已执行的作者实现；经GitHub检索确认了实际Affiliation库。AnomLLM notebook raw直连失败，未读到该notebook，不声称逐行复用了它的评价代码；方法依据来自已缓存全文。

## GNSS适配不能省略的说明

作者代码不接受无真值事件，且无预测时P为NaN。本项目保存原P的null状态，对完整正样本站汇总时将漏检站F1及汇总P置0；站均F1、平均P/R后调和F1、定义域内作者P均值均保存。无真值站不加满分，单列误报。

缺测后逐日mask会把长事件切成多个观测片段，可能明显影响Affiliation：同时报告保留语义跨度的敏感性版本，不压缩日期、不跨站拼接。距离概率仍在原历时域上积分，不能说缺测从作者积分中完全被排除了。点F1和事件IoU只用有效观测日；匹配边界MAE按匹配数加权，并列漏检数。

`range_union`会忽略“暂态还是形变”的类别差异。若暂态被数值误归形变但并集已经覆盖，类型修复对并集几乎没有提升；不得用类别指标的提升夸大成所有异常范围定位都显著改善。类型负站包含其他异常类型站，不等同于所有机制均正常的站；原通用任务正常站指标仍保留。

EXP21直接定位给出无数值候选的完整曲线/等价文字；文本采用round-trip精度，输入约8万tokens，远大于图像输入。它是信息内容对照，并非token预算相同的成本对照。若文本响应在正常前缀幻觉事件，被记录为契约失败，独立数值路径保留；这种回退产生的高分不能宣称来自文字理解。EXP22另用原短候选消息的同模型有图/去图对照，避免把超长文字输入问题当作视觉优势。
