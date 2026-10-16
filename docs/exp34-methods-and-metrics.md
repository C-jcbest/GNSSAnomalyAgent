# EXP34方法、视觉职责与指标对应

实验名称：**EXP34：视觉选区—去叙述证据判定（VDE-GNSS）**。
模型、数值拟合与数据生成机制保持原设置；视觉作用位置和最终系统收益分别评价。

| 报告方法键 | 真实流程 | 视觉负责什么 |
|---|---|---|
| `jcv_text_final_pure` | 图像粗筛→合并上下文→删除粗筛叙述→数值文字终判，仅模型暂态输出 | 根据原始N/E/U曲线选择检查范围，终判不看图 |
| `jcv_text_pipeline_pure` | 文字粗筛→自己的上下文→删除叙述→数值文字终判 | 全程去图；用于检验首轮视觉选区贡献 |
| `jcv_image_pure` | 图像粗筛→合并上下文→删除叙述→联合全局/局部图终判 | 前端选区和末端图像识别，用于检验终判再看图是否有效 |
| 上述去掉`_pure` | 模型confirmed暂态与近邻add-only融合 | 完整系统，检查视觉有无补回强数值漏报、增加误报或破坏其他类型 |
| `vector_nearest` | 原数值分段及向量近邻 | 没有视觉，本项目GNSS数值适配，未在本批训练或扫门槛 |
| 六原数值方法 | 原分段、森林及一般异常检测器 | 原库实现与既有GNSS适配边界沿用EXP28记录 |

保留旧方法键是为了精确对应复用的冻结输入构建，不代表本轮仍传递粗筛解释。
实际入口为`scripts/experiment_exp34.py`，评分为`scripts/report_exp34.py`。
`pure`只表示未融合近邻最终预测，仍输入数值阶跃软证据，不能叫完全纯视觉。
三种终判使用同一提示，但图/文字不等token预算，文字/图像粗筛可能选择不同区域；
该区域差异是首轮模态作用的一部分，不声称终轮匹配同一区域。

## 指标与报告文件

| 文件及字段 | 评价对象和解释 |
|---|---|
| `comparison.csv/transient_aff_f1` | 暂态异常范围定位，逐站Aff-F1宏平均，仅含有真暂态站 |
| `comparison.csv/deformation_aff_f1` | 持续位移活动范围定位；完整系统沿用数值，不能归为新增视觉能力 |
| `comparison.csv/range_union_aff_f1` | 暂态与形变并集定位，忽略这两类之间的分类错误 |
| `comparison.csv/range_types_aff_macro_f1` | 暂态和形变两类Aff-F1平均；不能与范围并集混称 |
| `range-metrics.csv` | 上述三种范围的Aff-P/R/F1及日P/R/F1/TP/FP/FN，含两种宏聚合读出 |
| `comparison.csv/transient_daily_f1` | 全站观测暂态日级微F1，惩罚无暂态站误报；原缺测日不当正常观测 |
| `comparison.csv/four_type_daily_f1` | 四类日F1宏平均；只输出暂态的pure不能与完整系统公平排名 |
| `summary.json/methods/*/range/*/negative_station_fp_days` | 该类无真值站的误报日，Aff主值未覆盖的部分必须另看 |
| `summary.json/methods/*/original`与`generic_affiliation` | 不区分类别的一般异常指标，与分类型评价不能互替 |
| `strata.json` | 种子/六场景分层，不能把开发分层当现场或独立测试 |
| `paired-changes.json` | 视觉/文字粗筛、终判图/文字、融合/近邻的逐站TP/FP变化及描述性bootstrap |
| `process.json` | 粗筛命中与失败，属于过程证据，不替代最终检测效果 |
| `costs-by-mode.json` | 真实请求、图像输入次数、token及失败，复用不重复计成本 |
| `admission.json` | 原登记的机制确认与系统升级，分开判断，不因某一个指标较高改变条件 |
| `cases/` | 最大新增TP、最大TP损失、最大新增FP和仅Aff改善的正反案例 |

范围评价执行保存的原作者Affiliation源码，聚合与缺测适配明确披露，无point adjustment。
没有执行TSAD-Agents等完整作者框架，本轮方法不署为其复现分数。
具体判定与数据隔离见[预登记](experiment-exp34.md)，文献读取边界见[EXP32文献记录](exp32-reading-notes.md)。
