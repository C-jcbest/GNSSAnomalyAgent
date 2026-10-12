# EXP02开发记录与冻结选择

仅使用新种子2026101203的验证数据；以下不是测试结果。固定首6站视觉小批按清单顺序选择，未挑选高分案例。

| 批次 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |
|---|---|---:|---:|---:|---:|---:|---:|
| exp02-numerical-validation | random_forest | 0.6114 | 0.5904 | 0.0968 | 0.2222 | 0.0000 | 0/24 |
| exp02-numerical-validation | rf_balanced | 0.6438 | 0.5988 | 0.0353 | 0.6667 | 0.0000 | 0/24 |
| exp02-numerical-validation | rf_gap | 0.6438 | 0.5988 | 0.0523 | 0.6667 | 0.0000 | 0/24 |
| exp02-numerical-validation | rf_temporal | 0.7360 | 0.6964 | 0.4762 | 0.6667 | 0.0000 | 0/24 |
| exp02-numerical-validation | extra_trees | 0.5107 | 0.4936 | 0.1304 | 0.2222 | 0.0000 | 0/24 |
| exp02-numerical-validation | isolation_forest | 0.4114 | — | — | — | — | 0/24 |
| exp02-independent-validation-b | rf_independent | 0.5481 | 0.5451 | 0.1154 | 0.2222 | 0.0000 | 0/24 |
| exp02-independent-validation-b | rf_independent_temporal | 0.4878 | 0.5046 | 0.6667 | 0.2222 | 0.0000 | 0/24 |
| exp02-morphology-validation | rf_morphology | 0.6674 | 0.5861 | 0.0420 | 0.6667 | 0.0000 | 0/24 |
| exp02-morphology-validation | rf_morphology_temporal | 0.7279 | 0.6533 | 0.4762 | 0.6667 | 0.0000 | 0/24 |
| exp02-small-v1 | guided_fusion | 0.7430 | 0.6855 | 0.5000 | 0.5714 | 0.0000 | 0/6 |
| exp02-small-v1 | guided_no_local | 0.8159 | 0.5448 | 0.5000 | 0.8000 | 0.0000 | 1/6 |
| exp02-small-v1 | guided_no_scores | 0.3639 | 0.3001 | 0.3333 | 0.5714 | 0.0000 | 0/6 |
| exp02-small-v1 | rf_temporal | 0.8416 | 0.8467 | 0.4000 | 1.0000 | 0.0000 | 0/6 |
| exp02-small-v2 | guided_fusion | 0.3418 | 0.2749 | 0.2500 | 0.6667 | 0.0000 | 0/6 |
| exp02-small-v2 | guided_no_local | 0.4176 | 0.3364 | 0.2857 | 0.8000 | 0.0000 | 1/6 |
| exp02-small-v2 | guided_no_scores | 0.3617 | 0.2887 | 0.4000 | 0.2857 | 0.0000 | 0/6 |
| exp02-small-v2 | rf_split_temporal | 0.8416 | 0.8467 | 0.4000 | 1.0000 | 0.0000 | 0/6 |
| exp02-anchored-development | anchored_fusion | 0.8757 | 0.8467 | 0.4000 | 1.0000 | 0.0000 | 0/6 |
| exp02-anchored-development | anchored_no_local | 0.8757 | 0.8467 | 0.4000 | 1.0000 | 0.0000 | 1/6 |
| exp02-anchored-development | anchored_no_scores | 0.7554 | 0.6984 | 0.6667 | 0.6667 | 0.0000 | 0/6 |

选择依据：时间解码改善持续活动的碎片问题；短窗特征改善尖峰但损害持续活动，按点/区间分配；独立二元RF不采用；自由视觉重写可能产生假活动，主方法改为同响应的数值锚定视觉复核。

v1、v2两批各18次模型请求；v3规则在已保存v2验证响应上离线开发，无新增调用。测试前冻结主方法anchored_fusion及相同融合规则的信息组消融。

失败开发运行exp02-independent-validation在训练参数写盘时TypeError（RandomForestClassifier对象不可JSON序列化）停止，无请求/预测；修正记录代码后新目录exp02-independent-validation-b执行，未覆盖。

较早的数值开发运行在单个进程执行期间源码又增加了后续视觉模块；预测使用进程已经加载的冻结版本，evaluation_source_hashes可能反映更晚磁盘源码。这些仅作为开发诊断；正式主批和复测期间不修改src，登记快照与运行源码一致。

完整预先声明规则见experiment-exp02.md；所有原始批次和失败保留，开发中查看过的验证集不冒充未见测试。
