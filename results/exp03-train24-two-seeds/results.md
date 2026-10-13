# GNSS-VLM-EXP03-v2：GNSS排序候选视觉确认、可修正活动边界与训练规模对照实验

范围：paired-train24-test-control。主方法ranked_boundary_fusion，旧主方法anchored_fusion；相同视觉响应派生，非独立调用。

| 批次 | 训练站 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| exp03-train24-seed2026101207-v2 | 24 | random_forest | 0.5445 | 0.4965 | 0.0000 | 0.0000 | 0.0000 | 0/24 |
| exp03-train24-seed2026101207-v2 | 24 | rf_balanced | 0.5519 | 0.4834 | 0.0000 | 0.2222 | 0.1667 | 0/24 |
| exp03-train24-seed2026101207-v2 | 24 | rf_temporal | 0.5574 | 0.4769 | 0.2857 | 0.2222 | 0.1667 | 0/24 |
| exp03-train24-seed2026101207-v2 | 24 | rf_split_temporal | 0.5574 | 0.4769 | 0.2857 | 0.2222 | 0.3333 | 0/24 |
| exp03-train24-seed2026101207-v2 | 24 | guided_fusion | 0.6059 | 0.5547 | 0.5556 | 0.5263 | 0.3529 | 1/24 |
| exp03-train24-seed2026101207-v2 | 24 | anchored_fusion | 0.6118 | 0.5589 | 0.2667 | 0.2222 | 0.3333 | 1/24 |
| exp03-train24-seed2026101207-v2 | 24 | boundary_fusion | 0.7261 | 0.6830 | 0.6667 | 0.2222 | 0.3333 | 1/24 |
| exp03-train24-seed2026101207-v2 | 24 | boundary_numeric_points | 0.7261 | 0.6830 | 0.6667 | 0.2222 | 0.3333 | 1/24 |
| exp03-train24-seed2026101207-v2 | 24 | ranked_boundary_fusion | 0.7267 | 0.6830 | 0.6667 | 0.6667 | 0.4615 | 1/24 |
| exp03-train24-seed2026101208-v2 | 24 | random_forest | 0.6144 | 0.6011 | 0.0533 | 0.0000 | 0.0000 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | rf_balanced | 0.6271 | 0.5506 | 0.0241 | 0.8333 | 0.2500 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | rf_temporal | 0.6545 | 0.6278 | 0.5882 | 0.8333 | 0.2500 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | rf_split_temporal | 0.6529 | 0.6278 | 0.5882 | 0.4000 | 0.2500 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | guided_fusion | 0.6339 | 0.5807 | 0.5556 | 0.3750 | 0.5000 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | anchored_fusion | 0.7069 | 0.6730 | 0.7143 | 0.4444 | 0.2500 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | boundary_fusion | 0.6480 | 0.6077 | 0.5714 | 0.4444 | 0.2500 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | boundary_numeric_points | 0.6489 | 0.6077 | 0.5714 | 0.4000 | 0.2500 | 0/24 |
| exp03-train24-seed2026101208-v2 | 24 | ranked_boundary_fusion | 0.6474 | 0.6077 | 0.5714 | 0.4615 | 0.4000 | 0/24 |

## 成对差值

形变日micro-F1，种子内站点bootstrap 2000次；条件于已拟合模型。

- ranked_boundary_fusion − anchored_fusion：+0.0425，95%CI [-0.0357, 0.1217]。
- ranked_boundary_fusion − boundary_fusion：+0.0000，95%CI [0.0000, 0.0000]。
- ranked_boundary_fusion − guided_fusion：+0.0861，95%CI [-0.0529, 0.2324]。
- ranked_boundary_fusion − rf_split_temporal：+0.1160，95%CI [-0.0023, 0.2409]。
- ranked_boundary_fusion − random_forest：+0.1071，95%CI [0.0394, 0.1926]。
- ranked_boundary_fusion − rf_balanced：+0.1385，95%CI [0.0277, 0.2623]。
- ranked_boundary_fusion − boundary_numeric_points：+0.0000，95%CI [0.0000, 0.0000]。

## 两种子合并（共同方法）

| 方法 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 暂态日F1 |
|---|---:|---:|---:|---:|---:|
| anchored_fusion | 0.6085 | 0.4828 | 0.3333 | 0.3000 | 0.4110 |
| boundary_fusion | 0.6510 | 0.6207 | 0.3333 | 0.3000 | 0.4110 |
| boundary_numeric_points | 0.6510 | 0.6207 | 0.3158 | 0.3000 | 0.4110 |
| guided_fusion | 0.5649 | 0.5556 | 0.4314 | 0.4242 | 0.4110 |
| random_forest | 0.5439 | 0.0185 | 0.0000 | 0.0000 | 0.2305 |
| ranked_boundary_fusion | 0.6510 | 0.6207 | 0.5366 | 0.4286 | 0.4110 |
| rf_balanced | 0.5125 | 0.0103 | 0.5714 | 0.2000 | 0.2554 |
| rf_split_temporal | 0.5350 | 0.4211 | 0.3158 | 0.3000 | 0.2087 |
| rf_temporal | 0.5350 | 0.4211 | 0.5714 | 0.2000 | 0.2087 |

全部四类型、±3日辅助及正常误报见CSV；缺测排除，无point adjustment。
开发验证不能充当未见测试；同分布仿真不能证明现场效果。完整论文复现或未运行方法不填成绩。
