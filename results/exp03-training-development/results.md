# GNSS-VLM-EXP03-v1：GNSS可修正活动边界与训练事件规模对照实验

范围：validation-training-scale。主方法boundary_fusion，旧主方法anchored_fusion；相同视觉响应派生，非独立调用。

| 批次 | 训练站 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| exp03-numeric-train96-validation | 96 | random_forest | 0.4541 | 0.5244 | 0.0000 | 0.7273 | 0.0000 | 0/24 |
| exp03-numeric-train96-validation | 96 | rf_balanced | 0.4940 | 0.5211 | 0.0129 | 0.6667 | 0.3077 | 0/24 |
| exp03-numeric-train96-validation | 96 | rf_temporal | 0.4894 | 0.5832 | 0.5833 | 0.6667 | 0.3077 | 0/24 |
| exp03-numeric-train96-validation | 96 | rf_split_temporal | 0.4889 | 0.5832 | 0.5833 | 0.6000 | 0.3636 | 0/24 |

## 成对差值

形变日micro-F1，种子内站点bootstrap 2000次；条件于已拟合模型。


## 同站训练规模对照

- train24 rf_split_temporal：形变日F1 0.3096；阶跃日F1 0.3333。
- train96 rf_split_temporal：形变日F1 0.5832；阶跃日F1 0.3636。

全部四类型、±3日辅助及正常误报见CSV；缺测排除，无point adjustment。
开发验证不能充当未见测试；同分布仿真不能证明现场效果。完整论文复现或未运行方法不填成绩。
