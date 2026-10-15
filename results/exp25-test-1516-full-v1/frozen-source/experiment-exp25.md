# EXP25：SDR-GR全面同站测评与模块消融

2026-10-15，任何本轮测试评分/请求前登记。父版本4e5259c，备份标签
`backup/pre-exp25-comprehensive-20261015-4e5259c`。沿用ARS实验执行/验证工作流。

## 数据与冻结方法

主测评使用已生成但从未评分的seed2026101516 **test 48站**；96训练站，48验证站
曾用于EXP23—24开发，本轮不得再用test调整任何参数/提示/修复规则。全部48站保留，
不按类别、触发、成功或结果筛选。最初6个station_id仅检查执行流程，允许负结果，
完成后配置不变执行其余42站。若认证/服务持续不可用则记录中止，不隐式重试。
原EXP06的48站及1516 validation 48站可补充复算，明确开发/历史分析，不混作新测试。

主研究方案为 **SDR-GR-Flash**：原Flash SDR修复＋EXP23候选恢复＋EXP24缺测结构约束。
其余配方在开测前固定如下：

| 方法键 | 身份/比较含义 |
|---|---|
| random_forest / extra_trees / isolation_forest | sklearn原库默认判决＋共用GNSS特征/训练适配；IF仅通用任务 |
| rf_balanced | 特别标注class_weight=balanced，不冒称默认RF |
| rf_split_temporal | 既有训练侧形态分工/时序解码 |
| segment_numeric | 独立位移结构分解，无LLM |
| visual_only | qwen3.8-flash纯视觉完整四类输出；去数值结构对照 |
| visualtime_gnss | 锁定VisualTimeAnomaly作者原函数，N/E/U×point/range；仅通用任务 |
| sdr_flash / sdr_vl | 原SDR流程，Flash/VL同站新调用有图 |
| sdr_flash_noimage / sdr_vl_noimage | 同模型同提示只删除image项，原SDR全图像移除 |
| sdr_raw | 原pair_raw修复，去慢轨迹扣除、保留正常历史残差，Flash |
| sdr_gr_flash / sdr_gr_vl | 同Flash原SDR基底，恢复复核分别Flash/VL，带缺测约束 |
| sdr_cr_flash / sdr_cr_vl | 同一恢复响应，去缺测约束 |
| sdr_gr_recovery_noimage | 原Flash有图基底，只去第二阶段恢复图像，单模块消融 |
| sdr_gr_pipeline_noimage | 第一阶段Flash去图，依其自身候选做去图恢复，完整去视觉消融 |
| sdr_gr_numeric_recovery | Flash基底，第二阶段换原returning_pair数值判别，非完整去LLM |
| numeric_gr | 从segment_numeric出发，同候选数值判别＋缺测约束，完整去LLM |
| sdr_gr_candidate_all | Flash基底，所有符合缺测结构的恢复候选直接接受，简单先验对照 |

不改src生产源码，不改原冻结EXP23/24提示/边界/窗口/信息准则，不设置跨站毫米/速率阈值。
第二阶段所有有图/去图配对以同一Flash有图基底生成候选。完整去图路径候选若不同，
额外独立请求；若其无图请求逐字相同可复用同一响应，明确共享，不虚计请求。
GR约束在响应后执行，保持EXP24实际测评配方；不假装筛选后的新提示已经调用过。
无候选站不请求，失败保留独立数值结果并单列依赖失败。纯视觉失败按零检测。
作者适配有部分子请求失败时保留已成功子请求的实际并集，并另报整站零检测敏感性，
不把失败站从总体删除。

## 原码与适配披露

- VisualTimeAnomaly锁定`7158c4ff05bc6fb27bf5045a1e8129318ca4a47d`，
  调用作者原create_openai_request/create_vision_messages，temperature=.4及原stop不改。
  修改仅为GNSS单轴图、3轴两任务并集、授权Flash模型、严格解析/无重试，非论文成绩复现。
  没有作者四类标签，不伪造其四类或形变成绩。
- ruptures 1.1.10 PELT作者库原码：API默认l2构造＋README pen10，以及README rbf/pen10。
  仅对比非尖峰形态变点边缘的严格/±3日P/R/F1，原库没有四类输出。
  pen10是README给定值，不是GNSS自动默认；GNSS结构适配的min_size、jump、代价和惩罚另列。
- Affiliation继续用Huet等锁定作者原码，inclusive→half-open、缺测及正样本站均值按EXP20。
- VLM4TS、TimeRCD等本轮未完整运行则不填分数，不用本项目模块冒充其原方法。

## 评价与统计

主指标：暂态、长期形变、两类并集的Affiliation P/R/站均F1；另报两类等权Aff-F1，
平均P/R后调和F1、缺测语义跨度敏感性。无GT站Aff=null，误报单列。
辅助：每类逐日P/R/F1、四类等权日F1、通用日P/R/F1与通用Affiliation，
事件F1(.3/.5，补充)、尖峰/阶跃严格和±3日、覆盖、最佳IoU、匹配边界起止MAE、
类型负站误报日/区间/站年、正常控制误报、活动漏日/停止后假活动日。
AP仅对已有连续数值分数报告，不给LLM二值输出伪造连续评分。
全部按同站、同真值、同缺测计算，不point-adjust，不扫测试阈值，不按IoU否决方案。

分6种模拟场景另列计数和指标。描述性配对站bootstrap固定seed2026101525、2000次，
报告主方案减数值/去图/去恢复的Aff-F1与四类日F1差值区间，不以显著性作为成败门槛。
所有站共享训练模型，区间不当作真实GNSS总体推断。
调用成本按实际请求/失败/token/中位请求时间，算法派生结果的依赖共享单列，
不将同一响应的多个消融版本重复计入总调用数。token缺失不补造。

结果发布包括多项主表、对比表、模块消融表、场景表、逐站原始计数、来源/适配表及图。
无论结果正负均报告；GR若没有干预机会明确标注。测试后不根据成绩切换主方法。
