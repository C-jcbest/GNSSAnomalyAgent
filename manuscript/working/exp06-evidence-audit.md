# EXP06 论文证据独立复核

2026-10-14。主方法 SDR-VLM=`pair_focus_repair`，主实验 GNSS-VLM-EXP06-v1。仅读取已保存输入、真值与预测；未发模型请求。

从两个种子 2026101405、2026101406 共 48 站的 672 份原始输出重算评价，与保存的逐站和汇总结果完全一致；复算全部 5 组 5000 次配对 bootstrap、活动误差、PELT 边缘匹配。每种子 96 训练、24 验证、24 测试，实际已核对 manifest。

## 主表（同组 48 站）

| 方法键 | 四类等权日 F1 | 四类事件 F1 IoU≥0.5 | 通用日 F1 |
|---|---:|---:|---:|
| random_forest | 0.2723087012 | 0.1249212776 | 0.5012475742 |
| rf_balanced | 0.4330308464 | 0.2493302731 | 0.5111856823 |
| extra_trees | 0.2209121339 | 0.0950801548 | 0.3933581449 |
| isolation_forest | — | — | 0.3710440160 |
| rf_split_temporal | 0.4187539702 | 0.3715296198 | 0.5459522878 |
| segment_numeric | 0.8322588320 | 0.8223739496 | 0.9140566274 |
| visual_only | 0.1138109158 | 0.1337126600 | 0.3486312399 |
| guided_fusion | 0.4650257613 | 0.4299171357 | 0.5549649160 |
| morphology_fusion | 0.5671897180 | 0.5011523752 | 0.6303546099 |
| localized_fusion | 0.5817399826 | 0.5157026398 | 0.6301758366 |
| pair_focus_repair | 0.8652251063 | 0.8522770398 | 0.9178116413 |
| pair_raw_repair | 0.8622101597 | 0.8522770398 | 0.9215442092 |

主方法较数值消融 +0.0329662743，较旧完整定位融合 +0.2834851238。前者主要对应返回对修复；后者主体提升来自位移结构分解。配对子任务 pair_focus/pair_raw 不是完整检测方法，不列入四类排名。

## 类型计数与修复

| 方法 | 类型 | TP | FP | FN | 日 F1 |
|---|---|---:|---:|---:|---:|
| segment_numeric | spike | 10 | 0 | 5 | 0.8000000000 |
| segment_numeric | step | 15 | 9 | 1 | 0.7500000000 |
| segment_numeric | transient | 446 | 0 | 123 | 0.8788177340 |
| segment_numeric | deformation | 1448 | 56 | 265 | 0.9002175940 |
| pair_focus_repair | spike | 10 | 0 | 5 | 0.8000000000 |
| pair_focus_repair | step | 14 | 4 | 2 | 0.8235294118 |
| pair_focus_repair | transient | 507 | 6 | 62 | 0.9371534196 |
| pair_focus_repair | deformation | 1448 | 56 | 265 | 0.9002175940 |
| pair_raw_repair | spike | 10 | 0 | 5 | 0.8000000000 |
| pair_raw_repair | step | 14 | 4 | 2 | 0.8235294118 |
| pair_raw_repair | transient | 494 | 5 | 75 | 0.9250936330 |
| pair_raw_repair | deformation | 1448 | 56 | 265 | 0.9002175940 |

48 站共有 15 个可观测尖峰、16 个阶跃、16 个暂态、16 个活动事件；暂态 569 日，活动 1713 日。主方法尖峰与活动结果完全等同数值消融。

正式发生 3 次修复：2026101405/s0065 恢复暂态 593—627，但删除同日起始的真实 593 阶跃；2026101406/s0049 把真实尖峰误配为 517—522 暂态，新增 6 个暂态假日；2026101406/s0071 恢复 623—651 暂态，真实 624—651，早 1 日，623 日另有真实尖峰。

阶跃计数 15/9/1 → 14/4/2，体现减少误报并损失一次召回；暂态日计数 446/0/123 → 507/6/62，减少 61 漏日并增加 6 假日。

## 活动范围和正常误报

真活动站仅 16 个。旧完整定位融合漏 944 日、停止后假活动 124 日、起点前 31 日；SDR-VLM 漏 265 日、停止后 6 日、起点前 7 日。所有 48 站活动 FP 为 56 日，不能将前后 13 日写成全部活动误报；另有 43 日发生在没有真实活动的站点。

8 个机制正常站中 1 个（2026101406/s0060）有 15 日假活动，另 7 个零误报。评价字段 normal_stations=9 包括 1 个尖峰日正好缺测的站点；不要将 9 与机制正常 8 混为一谈。

## 两种子与描述性区间

| 测试种子 | SDR-VLM | 数值消融 | 旧完整定位 |
|---|---:|---:|---:|
| 2026101405 | 0.8107827421 | 0.7976912199 | 0.5271165312 |
| 2026101406 | 0.9059851670 | 0.8510371412 | 0.6134645310 |

按 48 个站点有放回抽取 48 个配对站点，重复 5000 次，每次重新汇总四类 TP/FP/FN 后等权平均；种子 2026101404，2.5%/97.5% 分位数。不是站点 F1 的简单平均，不是种子分层 bootstrap。站点共享训练模型，区间用于描述性不确定性。

| SDR-VLM 减对照 | 差值 | 95% 区间 |
|---|---:|---|
| localized_fusion | 0.2834851238 | [0.1803558979, 0.4024676904] |
| morphology_fusion | 0.2980353883 | [0.2062395973, 0.4100089258] |
| segment_numeric | 0.0329662743 | [0.0000000000, 0.0777273891] |
| rf_split_temporal | 0.4464711361 | [0.3583600720, 0.5351958101] |
| pair_raw_repair | 0.0030149467 | [-0.0454027711, 0.0533094081] |

## 作者 PELT 辅助任务

真值取非尖峰事件 start；暂态、活动再取 end+1（返回或停止状态第一天），同日边缘去重。仅真边缘当天与前一天都观测才纳入，预测边缘要求当天观测；所有方法共用该口径。主匹配严格同日，辅助匹配 ±3 日，一对一最大匹配。过滤后 77 个真边缘。

| 方法 | 严格 TP/FP/FN | 严格 F1 | ±3 日 TP/FP/FN | ±3 日 F1 |
|---|---|---:|---|---:|
| author_api_defaults_l2_pen10 | 11/258/66 | 0.0635838150 | 52/217/25 | 0.3005780347 |
| author_readme_rbf_pen10 | 8/45/69 | 0.1230769231 | 38/15/39 | 0.5846153846 |
| gnss_displacement_structure_adapter | 48/35/29 | 0.6000000000 | 59/24/18 | 0.7375000000 |

PELT 原作者搜索代码未改；l2 默认构造 min_size=2、jump=5，pen=10 来自 README 示例；rbf 示例亦 pen=10。GNSS 适配是三响应线性代价、min_size=7、jump=1、6log(3n)。所有方法均共用正常参考残差标准化和缺测时间映射。PELT 不输出四类标签，不能替它填写四类日 F1。

## 请求与失败

开发 76 次请求，正式 110 次，共 186；正式为纯视觉 48（失败 1）、旧完整图 48（失败 0）、主配对输入 7（失败 0）、图像消融 7（失败 0）。主方法 48 站中 7 站发请求，41 站不请求；主方法派生行 request=0 表示修复无新增请求，方法实际成本应记父方法的 7 次。纯视觉失败按零检测；修复失败保留独立数值判决。无隐式重试。

## 后续扩展批次（独立列示）

| 批次 | 站数 | SDR-VLM | 数值消融 | 增量 |
|---|---:|---:|---:|---:|
| exp07 | 48 | 0.8665421343 | 0.8109944277 | 0.0555477066 |
| exp08 | 48 | 0.7724288634 | 0.7387210963 | 0.0337077671 |

已核对 EXP07/08 四个 baseline 注册的 23 个生产源码哈希与 EXP06 一致，并核对冻结文件实际哈希；软件环境相同。EXP08 分离数值与视觉运行，original 路径调用同一 prepare_pair_focus、SYSTEM 与 repair_return_pairs；四批全部请求的模型、温度、thinking、响应 schema 和 system prompt 与 EXP06 一致。EXP07/08 主方法和数值消融共 192 份预测再从真值重算，与保存逐站分数一致。

可作为新增种子的扩展复核，需同时列出较低的 EXP08 .7724；不同协议文件哈希来自新批次登记，不代表 SDR-VLM 算法改动。不得写成原预登记 144 站，不直接平均各批 F1。EXP09 为开发，未纳本报告。

## 文本与代码不一致点

- Imputation：training.json says median; frozen baselines.py calls SimpleImputer() Write training-only mean imputation.
- Spike RF：48-feature rf_morphology uses class_weight=balanced Label balanced RF adaptation; default predict rule does not mean default class weighting.
- PELT edges：Truth requires edge and previous day observed; prediction requires edge observed. End edge is end+1; coincident edges deduplicated. Disclose exact auxiliary task.
- Visual cost：pair_focus_repair requests=0 but parent pair_focus requests=7 Report SDR-VLM total 7 visual requests.
- Activity false days：activity-errors only true activity stations; overall FP=56, before+after=13 Remaining 43 false activity days belong to non-active stations.
- Failure policy：base-summary generic prose says all failure zero; final repair preserves numeric decisions Use final report: pure visual fails empty; repair fails keeps numeric.
- PELT penalty metadata：summary parameters numerical penalty is copied from final station Report 6log(3n), not universal 43.1666.

## 来源

- results/exp06-main-v1/{summary.json,per-station.json,paired-bootstrap.json,activity-errors.json,run-audit.json}
- results/exp06-pelt-main-v1/{summary.json,per-station.json}
- runs/exp06-seed2026101405-v1 与 exp06-seed2026101406-v1 的 registration/source/predictions
- data/exp06-seed2026101405 与 exp06-seed2026101406 的 observations/truth/manifest.json
- scripts/compare_exp06_pelt.py、scripts/summarize_exp06.py、src/gnss_research/metrics.py
- results/exp07-main-v1 与 exp08-main-v1，及各登记正式 baseline/visual 运行

完整高精度指标、类型事件计数、成本 token、案例原始事件与扩展逐种子结果见同名 JSON。

## 扩展逐种子与案例补充

| 批次/种子 | SDR-VLM | 数值消融 |
|---|---:|---:|
| exp07/2026101408 | 0.8575069509 | 0.7723421157 |
| exp07/2026101409 | 0.8723190978 | 0.8471383987 |
| exp08/2026101412 | 0.7181900992 | 0.7282938189 |
| exp08/2026101413 | 0.8207917144 | 0.7518394896 |

EXP08 的 2026101412 种子视觉负增量约 -0.0101；不能写全部 6 个种子都获益。EXP07/08 没有旧完整定位 localized_fusion 的同组执行成绩，不纳入扩展表。旧数值 rf_split_temporal 在两批为 .4685363121/.4227023704，其身份不同。

活动案例：2026101405/s0052 真值 541—670，旧完整定位 510—643，数值与 SDR-VLM 541—669；2026101406/s0052 真值 446—545，旧完整定位无输出，数值与 SDR-VLM 449—542。

暂态事件 IoU≥.5（≥.3 相同）：数值 TP/FP/FN=12/0/4、F1=.8571428571；SDR-VLM=14/1/2、F1=.9032258065。此为事件统计，不能与暂态日 F1=.8788177340/.9371534196 混淆。
