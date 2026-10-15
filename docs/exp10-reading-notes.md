# EXP10：视觉贡献定向调研与方法归属

2026-10-15。本轮重点是图像上下文、示例及判别职责，不用提高数值分数代替视觉贡献。

| 来源/实读范围 | 用于本轮的方法依据 | 证据边界 |
|---|---|---|
| Xu等，*Can Multimodal LLMs Perform Time Series Anomaly Detection?*，本轮新取得arXiv:2502.17812v1全文并重读§3.3–3.4、§4与作者提示代码 | 多变量按子图组织；点型、区间型、变量型任务分开，说明输入结构与任务粒度会影响视觉表现。本轮让图像模型负责共存水平形态而非稀疏点型数值定位 | 这篇是VisualTimeAnomaly，不是AnomLLM。作者库README声称WWW'25，本轮Crossref精确同题/同作者记录DOI10.1145/3774904.3792376，出版2026-04-12、ACM Web Conference2026；书目信息按新记录注明，不把旧README年份当成最终会议信息 |
| He等，*Harnessing Vision-Language Models for Time Series Anomaly Detection*，AAAI2026，DOI10.1609/aaai.v40i26.39319；已核正式摘要，既有方法全文/作者代码 | 轻量定位与全局上下文复核分工。本轮固定数值定位，改变视觉判别输入与输出权限，直接检验增量 | 未完整执行ViT4TS/VLM4TS，不把本项目数值候选称ViT4TS复现，不把作者24.6%提升移作GNSS成绩 |
| Zhou、Yu，AnomLLM，ICLR2025；既有全文，本轮重读§4.2、图像输入和结论 | 图像通常优于文字、显式CoT没有稳定帮助、模型间差异大；据此保留单轮、增加真正的图像消融 | 不将“多给推理步骤”当创新，也不把TAMA或VisualTimeAnomaly与此论文合并引用 |
| TAMA，*See it, Think it, Sorted*，arXiv:2411.02465v1；重读参考学习、多尺度自反思及聚合 | 参考图帮助捕获形态，局部放大使模型获得新视觉信息；本轮训练形态示例和统一上下文为待测适配 | 原文参考学习主要给正常片段，本轮带标签异常/共存示例是新增适配；Crossref题名搜索未返回精确TAMA记录，不据近似结果猜测正式出版信息 |

新原文、Crossref检索原响应和SHA清单保存在`papers/exp10-visual-evidence/`。
作者提示代码既有缓存`papers/evidence/visualtime_src_prompt.py`，只读未修改；
该原提示输出点/区间/变量ID，本轮额外GNSS竞争配对图、训练参考、限定输出结构为项目方法，
明确不是作者默认。没有改变作者源码后冒充无适配复现。

设计推论是：把视觉任务从重复看许多相似候选图改成在统一上下文上判断永久/返回关系，
并以固定训练形态参考降低共存歧义。是否有效必须通过新的同站实验和无视觉/无图/无参考消融验证。
原数值对照完整保持，不能用形态名称、训练标签的文字或更多候选本身冒充图像作用。

## EXP11—15补充实读与设计边界

本轮进一步实读TAMA全文Multimodal Reference Learning、Multimodal Analyzing、
Multi-scaled Self-reflection和Results Aggregation四段：其参考主要是正常图像，
复核给出局部放大图避免同图重复；最终用置信累积与阈值聚合，其表1含point-adjusted F1。
本项目不照搬该聚合阈值或point adjustment，亦不将带异常标签训练示例称作者默认。

重读AnomLLM§4.2.1—4.2.2及§5.1：作者用Matplotlib图像与多种数字文字表示、
结构化区间输出，并比较四类多模态模型。模型与输入形式应当分别控制；
因此EXP12/13仅改模型，EXP15又增加相同局部向量的JSON文字对照。
本文只能借其研究设计，不能借用其affiliation F1数值或不同数据上的模型排名。

实读作者缓存vlm4ts_src_models_vlm4ts.py构造器、detect与verify_intervals说明：
作者为ViT4TS筛选＋VLM核验，默认alpha=.01、vlm_model=gpt-4o；
preprocess.py的preprocess_time_series先detrend再min-max。
这些是作者源码事实，本项目不把PELT/RF候选称原ViT4TS，也未声称完整执行该作者模型。

官方Qwen视觉文档确认qwen3-vl-plus图像问答与enable_thinking=false支持，并建议
一般非工具视觉对话不设置system message。EXP12/13为隔离模型因素保留既有system组织；
未把该建议混入模型对照。代理只提供名称与返回模型信息，后端权重身份无法独立验证。
另查得2026年On Cost-Effectiveness of Language Models for Time Series Anomaly Detection
元数据（DOI10.3390/info17010072），出版社全文返回403；没有读到全文，不能引用其方法结论。
检索和访问范围保存在model-and-search-provenance.json。

EXP14局部曲线对齐仍失败；4次简单上/下阶跃图输入诊断两模型各2/2正确，
这只排除“所有图片都未被读取”的粗略怀疑，不证明能理解复杂GNSS图。
EXP15据此简化为局部三分量跳变量图，数值边界与已有活动/暂态不动；
局部中位数、7日窗、MAD表示及近邻配对都是项目适配，必须与视觉判断分列。
其数字文字对照提供相同完整统计；图像版还包含原始曲线上下文，
差异不能只归因条形图本身。简单近邻若同样优秀，必须披露，不能宣称视觉不可替代。
