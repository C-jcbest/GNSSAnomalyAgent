# GNSS-VLM-EXP03-v2：GNSS排序候选视觉确认、可修正活动边界与训练规模对照实验

范围：test-seed2026101208。主方法ranked_boundary_fusion，旧主方法anchored_fusion；相同视觉响应派生，非独立调用。

| 批次 | 训练站 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| exp03-seed2026101208-v2 | 96 | random_forest | 0.6699 | 0.6489 | 0.0196 | 0.4444 | 0.0000 | 0/24 |
| exp03-seed2026101208-v2 | 96 | rf_balanced | 0.5393 | 0.4540 | 0.0063 | 0.6154 | 0.2500 | 0/24 |
| exp03-seed2026101208-v2 | 96 | extra_trees | 0.6509 | 0.6342 | 0.0250 | 0.4444 | 0.0000 | 0/24 |
| exp03-seed2026101208-v2 | 96 | isolation_forest | 0.4495 | — | — | — | — | 0/24 |
| exp03-seed2026101208-v2 | 96 | rf_temporal | 0.5651 | 0.4892 | 0.6957 | 0.6154 | 0.2500 | 0/24 |
| exp03-seed2026101208-v2 | 96 | rf_split_temporal | 0.5649 | 0.4892 | 0.6957 | 0.6667 | 0.0000 | 0/24 |
| exp03-seed2026101208-v2 | 96 | fusion | 0.2918 | 0.2233 | 0.1429 | 0.2000 | 0.0000 | 1/24 |
| exp03-seed2026101208-v2 | 96 | guided_fusion | 0.7268 | 0.6900 | 0.6667 | 0.3750 | 0.6154 | 0/24 |
| exp03-seed2026101208-v2 | 96 | anchored_fusion | 0.7252 | 0.7006 | 0.8750 | 0.6667 | 0.0000 | 0/24 |
| exp03-seed2026101208-v2 | 96 | boundary_fusion | 0.7743 | 0.7661 | 0.7500 | 0.6667 | 0.0000 | 0/24 |
| exp03-seed2026101208-v2 | 96 | boundary_numeric_points | 0.7743 | 0.7661 | 0.7500 | 0.6667 | 0.0000 | 0/24 |
| exp03-seed2026101208-v2 | 96 | ranked_boundary_fusion | 0.7703 | 0.7661 | 0.7500 | 0.4000 | 0.6154 | 0/24 |

## 成对差值

形变日micro-F1，种子内站点bootstrap 2000次；条件于已拟合模型。

- ranked_boundary_fusion − anchored_fusion：+0.0655，95%CI [-0.1388, 0.2322]。
- ranked_boundary_fusion − boundary_fusion：+0.0000，95%CI [0.0000, 0.0000]。
- ranked_boundary_fusion − guided_fusion：+0.0761，95%CI [0.0000, 0.2515]。
- ranked_boundary_fusion − rf_split_temporal：+0.2770，95%CI [0.0613, 0.4816]。
- ranked_boundary_fusion − random_forest：+0.1172，95%CI [0.0109, 0.2469]。
- ranked_boundary_fusion − rf_balanced：+0.3121，95%CI [0.1340, 0.5005]。
- ranked_boundary_fusion − fusion：+0.5428，95%CI [0.3676, 0.6872]。
- ranked_boundary_fusion − boundary_numeric_points：+0.0000，95%CI [0.0000, 0.0000]。

全部四类型、±3日辅助及正常误报见CSV；缺测排除，无point adjustment。
开发验证不能充当未见测试；同分布仿真不能证明现场效果。完整论文复现或未运行方法不填成绩。
