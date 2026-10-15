# EXP25方法身份与指标说明

本实验名为**SDR-GR全面同站测评与模块消融**，主方案固定为`sdr_gr_flash`。
本页解释结果表的比较范围；全部参数在测试前沿用原配置，无测试阈值搜索。

## 原方法、原库与GNSS适配

| 对照 | 实际执行与适配 | 不应如何表述 |
|---|---|---|
| RF、ExtraTrees | sklearn原分类器，除random_state外沿用默认构造；训练侧中位数填补、共同GNSS特征，原predict判决 | 不是原始位移直接输入，也不是特定近期论文的复现 |
| balanced RF | 另设class_weight=balanced，明确单列 | 不能混写为默认RF |
| Isolation Forest | 原库默认构造/predict，训练仅正常前缀，共用特征；仅通用任务 | 没有四类输出，不能填四类或形变分数 |
| rf_split_temporal | 项目已有特征分工、训练侧Viterbi解码 | 是本项目适配，不是作者独立新模型 |
| segment_numeric | RF尖峰＋原位移结构分解、形状比较与返回对；无LLM | PELT原库的使用不代表整体是作者原算法 |
| numeric_gr | 从segment_numeric出发，同候选数值返回判别＋缺测约束 | 是完整去LLM对照，不能与只替换第二阶段混同 |
| visual_only | Flash原始NEU图独立完整四类检测，原提示不改 | 不含结构候选，不能冒充VisualTimeAnomaly |
| visualtime_gnss | 锁定作者原请求/提示函数，三轴×点/区间六请求，模型换Flash，GNSS图，严格解析 | 是作者源码的GNSS适配，非原数据/原模型成绩复现 |
| visualtime_author_parser | 同一批响应，另用未修改作者parse_output和向量转换函数离线解析；0新增请求 | 是登记后的解析补充，不能隐去严格解析失败或称新模型 |
| PELT l2、rbf | ruptures 1.1.10原库；默认构造l2及README rbf/pen10 | pen10是README示例给值，不是自动GNSS阈值；只比变点边缘 |
| PELT GNSS适配 | 自定义三响应线性代价、min_size7、jump1、6log(3n)惩罚、原时间映射及±3日细化等 | 参数有变化，必须标适配；不是原库默认配置 |

RF完整参数/训练站清单见`runs/exp25-test-1516-baseline-v1/training.json`，
源码与哈希见同run的registration/source；原请求/提示与作者锁定信息见visual run。
结构惩罚、窗口、单活动区间限制沿用[EXP06参数说明](experiment-exp06.md)，
恢复候选与缺测约束见[EXP23—24](exp23-24-assessment.md)。没有跨站统一毫米或速率告警阈值，
但算法有时间窗口与信息准则，不能声称完全无参数。

## 消融能回答什么

`sdr_gr_pipeline_noimage`从第一阶段开始删图，并从其自身预测生成第二阶段候选；
是完整去图对照。`sdr_gr_recovery_noimage`保留第一阶段有图，只删除第二阶段恢复图。
`sdr_gr_numeric_recovery`同样保留第一阶段Flash，仅将第二阶段换数值；不是完整去LLM。
`sdr_flash`去候选恢复；`sdr_cr_flash`去缺测约束；`sdr_raw`去额外慢轨迹辅助图。
Flash/VL恢复共享Flash第一阶段基底，`sdr_vl`则是第一阶段也换VL，两者用途不同。
共享请求的不同派生版本不重复计入总调用，单方法依赖成本不能逐行相加。

## 评价口径

范围主评价为暂态、持续形变、二者并集的Affiliation P/R与**正样本站F1均值**；
另报宏P/R调和F1及缺测跨度敏感性。无GT站记null并另报误报，不把双空站加满分。
暂态误归形变可能改善类型指标而不改变并集覆盖，因此分类型与并集必须同时看。
通用检测Affiliation针对四类并集，与二类范围并集不同。

逐日指标无point adjustment；四类日F1是各类型汇总TP/FP/FN后F1的等权均值。
范围事件F1(.3/.5 IoU)、覆盖、最佳IoU和匹配边界MAE为辅助，不作为方案采纳硬门槛。
尖峰/阶跃另报严格及±3日P/R/F1。所有评分保留原时间轴并排除原始缺测标签；
Affiliation距离积分仍在原历时域上，不能称缺测日期从积分中完全删除。
活动漏日、起点前和停止后假活动日只统计有真实形变的站；正常站误报单列。
AP只使用既有连续数值评分，LLM二值输出不伪造连续分数。

2000次配对站bootstrap仅作描述性区间；同站共用拟合模型、模拟真值、单新种子均限制外推。
不以显著性决定成败，不按测试结果换主方案。作者部分子请求成功时保留实际并集，
整站零检测敏感性同时披露。HTTP成功与输出解析成功分列，主方案数值独立路径的
视觉依赖失败记录保留在原回执，不能只看最终派生status。

## 文献依据与已验证来源

- Huet等，KDD2022，[Local Evaluation of Time Series Anomaly Detection Algorithms](https://doi.org/10.1145/3534678.3539339)：Affiliation原码commit `8d8449858096bbade6a6e70848d05c9cc9b846fe`，本项目原码不改。
- [Can Multimodal LLMs Perform Time Series Anomaly Detection?](https://arxiv.org/html/2502.17812v1)，§3.4对点/范围任务采用Affiliation；[正式WWW2026记录](https://doi.org/10.1145/3774904.3792376)。作者源码锁定`7158c4ff05bc6fb27bf5045a1e8129318ca4a47d`。
- Zhou、Yu，[Can LLMs Understand Time Series Anomalies?](https://arxiv.org/html/2410.05440v3)，ICLR2025：离散输出用affinity F1。作者跨站双空处理与本项目不同，不能横比论文表格绝对分数。

全文实读定位、AnomLLM代码与本地证据见[已有读书笔记](exp20-reading-notes.md)及
[EXP23文献说明](exp23-24-assessment.md)。VLM4TS、TimeRCD本轮未完整运行，不填成绩或声称胜过它们。
