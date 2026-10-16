# EXP30方法、源码与指标对应

实验名称：**EXP30：视觉粗筛与自适应局部重定界（AIR-GNSS）**。
有效方法评估使用 **EXP30-S1三窗实现修正版**；S1不是根据检测分数挑选的算法变体。
原四窗实现、连接失败回退与S1分别留证，不能合并成同一次完整方法评估。

## 入口与身份

| 名称/结果键 | 实际流程 | 源码与适配身份 |
|---|---|---|
| AIR-GNSS / `air_image` | 三窗图像粗筛→局部图重新定界＋可错的数值阶跃证据→近邻互补 | 本项目GNSS适配；`scripts/experiment_exp30_s1.py`为实际入口，显式复用冻结的`experiment_exp30.py`并修正窗口生成 |
| `air_global` | 相同图像粗筛与区域，第二轮查看全局图 | 去局部放大的对照；匹配轮数与选区，不声称token预算完全相同 |
| `air_text_final` | 相同图像粗筛，局部阶段使用同一窗口的全部文字点 | 隔离最后一轮图像相对文字的贡献 |
| `air_text_pipeline` | 文字粗筛→自己的局部文字点→近邻互补 | 完整文字流程对照；两条路径可能选择不同窗口 |
| `air_image_no_numeric` | 相同图像粗筛和局部图，删除数值阶跃证据 | 数值软证据消融 |
| 以上各键＋`_pure` | 仅保留对应局部confirmed暂态，不加入数值结果 | 用于观察视觉检出和融合遮蔽；只输出暂态，不能以四类F1与完整系统公平排名 |
| `vrc_image` / `vrc_text` | EXP29原全局图/文字自主范围互补，在本次同站重跑 | 本项目旧方法；S1 pilot按请求SHA复用技术复测中完全相同的144个收据，含失败；不是新独立请求 |
| `vector_nearest` | 冻结数值阶跃及三轴向量近邻 | 本项目GNSS数值适配；复用EXP28独立96站模型，无本轮重训或阈值扫描 |
| 其余数值方法 | 沿用EXP28登记的六个数值对照 | 原方法与GNSS适配边界沿用EXP28记录；本轮同站评分 |

AnomaMind、TimeRLM和DAST提供流程设计参考。本轮未运行其完整作者框架，表中不填写这些论文的GNSS成绩。
作者代码阅读范围和全文来源见[文献记录](exp30-reading-notes.md)。

## 结果文件与解释

有效pilot输出为`results/exp30-validation-pilot-v1/`。只有满足登记门槛，才生成full开发及test报告。
未生成的报告不能视作已有结果。

| 文件/字段 | 含义与使用范围 |
|---|---|
| `comparison.csv` / `transient_aff_f1` | 暂态真阳性站的Affiliation F1宏平均；主范围指标，不代表负站误报已得到惩罚 |
| `range-metrics.csv` | 各类Affiliation P/R/F1及日级P/R/F1、TP/FP/FN；与主表并读 |
| `comparison.csv` / `transient_daily_f1` | 全站暂态日级微平均F1；按原始预测与缺测掩码统计，无point adjustment |
| `comparison.csv` / `four_type_daily_f1` | spike/jump/transient/deformation四类日级宏F1；只有完整系统可直接比较 |
| `summary.json` | 全部聚合指标，包括一般异常、长期形变及其他范围指标 |
| `strata.json` | 三种子与六场景分层结果；不能把开发分层当独立确认 |
| `paired-changes.json` | 主方法相对各对照的逐站TP/FP/Aff变化，以及描述性配对bootstrap |
| `process.json` | 粗筛疑点、阳性站粗筛命中与失败；属于过程证据，不是最终检测提升 |
| `contribution-audit.json` | 各局部路径confirmed/uncertain、近邻覆盖、新增区间、真实TP/FP和负站误报，来自冻结预测的事后审计 |
| `admission.json` | 原登记扩量/采纳条件逐项判定；不因某个指标较高而改写条件 |
| `request-receipts.json` | 当前有效评估依赖的收据；全轮失败、中止及复用费用另见最终归档成本 |
| `cases/cases.json`与PNG | 按最大新增TP、最大新增FP及仅Aff改善规则选择的正反案例，避免只展示成功站 |
| `cases-air_global/` | 对预登记全局重读对照使用相同正反选择规则的补充案例，不能冒充主方法结果 |

完整去图对照是`air_text_pipeline`，不是`air_text_final`；后者保留了图像粗筛。
暂态以外的类别沿用数值结果，本轮不据四类F1变化宣称长期形变获得了新的视觉识别能力。

## 当前实现的能力边界

粗筛至多两个区域、两侧各扩35日、固定两轮及三种子pilot/full门槛均为明确披露的本项目资源与流程设置，
不冒充参考论文默认参数。confirmed/uncertain只表示模型决策；模型文字解释仍须与曲线和真值核验。
两区域可能切断同一暂态的两端；相接上下文合并的离线可见性分析只用于提出后续假设，尚未形成新预测成绩。
