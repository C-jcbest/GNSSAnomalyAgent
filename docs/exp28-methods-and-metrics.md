# EXP28方法身份、扩大样本与指标口径

实验全称为**EXP28：竞争解释反事实视觉复核（CEV-GNSS）与扩大样本评估**。
预定主方法是`ce_flash_image`，不是本轮各项分数最高的方法。
原CFV、EXP27变体和原SDR在新站重新执行；开发、测试各360站，不能把它们与旧48站表直接相减当改进量。

## 22种实际执行方法

| 表中方法键 | 含义与适配身份 |
|---|---|
| `ce_flash_image` | 新CEV-Flash：原上下文图及逐候选四列竞争解释图，qwen3.8-flash |
| `ce_flash_cfonly` | 同CEV提示，仅原CFV图；隔离新增竞争解释图的贡献 |
| `ce_flash_raw` | 同CEV提示，仅原位移图 |
| `ce_flash_text` | 同CEV提示，用同窗口逐日归一化原位移、历史残差及起点/返回向量替代图 |
| `ce_flash_noimage` | 主请求仅删除图片，保留候选及其余文本；没有逐日观测，属于较弱控制 |
| `ce_vl_image`、`ce_vl_text`、`ce_vl_noimage` | 同CEV设计，模型替换为qwen3-vl-plus，三个版本分别计分 |
| `cf_flash_image` | EXP26冻结原CFV-Flash：原图与仅扣候选暂态后的图；本轮重新调用 |
| `cf_flash_text`、`cf_flash_raw`、`cf_flash_noimage` | EXP28-S1补充消融，复制原CFV请求，仅换图或删图，三个版本分别计分 |
| `edge_flash_image` | EXP27原ECV：原CFV加双边界放大图 |
| `trend_flash_image` | EXP27原TCV：以局部趋势分离估计offset并加边界图 |
| `sdr_flash` | 原SDR-Flash，沿用其RF来源候选和慢轨迹辅助图；候选与CFV/CEV不完全相同 |
| `vector_nearest` | 同自由候选的原数值向量近邻，无LLM、未调参；强数值对照 |
| `segment_numeric` | 项目原位移结构分解，RF尖峰与返回配对；无视觉，是直接数值基底 |
| `random_forest`、`extra_trees` | sklearn原分类器，random_state固定，共同GNSS特征和训练侧默认均值填补，原predict |
| `rf_balanced` | 另设class_weight=balanced的RF，明确为适配，与默认RF分列 |
| `rf_split_temporal` | 项目已有点/区间特征分工与训练侧时序解码，是适配方法 |
| `isolation_forest` | sklearn原库规则，仅正常参考前缀拟合、共同GNSS特征，仅输出通用异常 |

数值模型只在96独立训练站拟合一次，两评估分区使用同一模型文件。
sklearn版本、完整参数、站ID与模型SHA见`runs/exp28-shared-training-v1/training.json`。
核验保存的模型确认SimpleImputer策略为默认`mean`。既有训练回执的说明文字误写为median，
本轮保留原回执并作此纠正，以实际模型参数为准，未重新训练或改变填补行为；
模型核验记录见`results/exp28-validation-comprehensive-v1/training-adapter-clarification.json`。
RF、ExtraTrees和Isolation Forest是原库加披露的输入适配，不是某篇近期论文的完整复现。
结构分解调用ruptures原PELT，但自定义代价、惩罚与时间细化属于本项目适配，
详情沿用[EXP06参数](experiment-exp06.md)与[既有身份说明](exp25-methods-and-metrics.md)。
本轮没有新增阈值搜索或跨站统一毫米/速率告警阈值；既有时间窗口与结构参数仍存在，不称完全无参数。

## 样本如何扩大

总816个独立模拟站，96训练、360开发、360测试；每站730日，365日正常参考前缀。
开发和测试分别包含三个种子、每种子120站，各分区六场景每类60站。
研究生成脚本直接调用原`simulate`，仅解耦训练与评估数量，不复制旧站、筛选难例或复杂化噪声。
站ID、随机流和训练集合互斥；机制仿真真值不是现场实测或专家金标准。

每分区先60站验证流程，再完成300站。CEV开发分数不佳也继续完整测试，
以免只有小批正结果才扩量。CEV主方法、模型、提示、后处理在测试前冻结。
CFV匹配消融在完整开发之后、测试调用与评分之前追加登记，属于EXP28-S1，不能说是原主协议预计划。
原测试小批发生连接故障，技术复测与首次失败分别保留，见[技术复测说明](exp28-transport-replication.md)。

## 主要读哪些指标

范围主评价为**暂态Affiliation P/R及正样本站宏F1**，用Huet等作者锁定源码；
先对每个有该类型真值的站计算F1，再取平均。它不等于宏P/R的调和均值。
无该类真值的站不加满分，而以误报日及负站误报单列；全站仍参与日P/R/F1与四类指标。
日F1采用全站TP/FP/FN汇总，无point adjustment；四类日F1是四种类别各自汇总F1的等权平均。
覆盖、事件IoU F1、边界误差仅辅助，不以IoU领先作为接受条件。
缺测沿原时间轴评价并排除缺测标签，Affiliation距离积分仍在原历时域，不能称完全删除缺测时间。

因此，Affiliation改善可以支持范围邻近定位更好，但可能伴随更多误报或较低日覆盖。
甚至零重叠近邻区间也可能获得非零Affiliation分数，正反案例必须同时列TP/FP/FN。
RF等方法若密集输出，可能Aff较高而日精度低；保留其原分数，不因不利排名删掉它们。
Isolation Forest无类别输出，类型和范围项留空，不伪造类型成绩。

全站含失败回退结果是主表；双方请求均成功的配对子集仅用于识别图像差异是否由失败回退驱动。
无自由候选的站不请求，保留在总体；触发子集、混合场景和三种子各自报告，不能替代总体。
描述性配对站bootstrap不给显著性采纳门槛，共用训练模型、单一生成器与有限触发站数仍限制泛化。
长期形变活动和内部阶段沿用数值结果，视觉没有扩大输出权限，不能宣称新增长期形变收益。

## 文献与原源码边界

- Huet等，*Local Evaluation of Time Series Anomaly Detection Algorithms*，KDD2022，
  [DOI](https://doi.org/10.1145/3534678.3539339)。Affiliation原码锁定
  `8d8449858096bbade6a6e70848d05c9cc9b846fe`，不改作者公式；本项目提供缺测和跨站聚合适配。
- [Can Multimodal LLMs Perform Time Series Anomaly Detection?](https://arxiv.org/html/2502.17812v2)、
  [VLM4TS](https://arxiv.org/html/2506.06836v1)、
  [Set-of-Mark](https://arxiv.org/html/2310.11441v1)提供多粒度、候选核验和位置标记启发。
  原文、读取范围与SHA见[本轮读书笔记](exp28-reading-notes.md)。竞争解释图是本项目推导。

本轮没有在720站上完整执行VisualTimeAnomaly、VLM4TS或TAMA作者管线。
旧48站的作者适配成绩不能搬入本轮同站排名，也不能由本轮胜过RF/数值近邻推断胜过这些论文方法。
