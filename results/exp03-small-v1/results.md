# GNSS-VLM-EXP03-v1：GNSS可修正活动边界与训练事件规模对照实验

范围：validation-six-stations。主方法boundary_fusion，旧主方法anchored_fusion；相同视觉响应派生，非独立调用。

| 批次 | 训练站 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| exp03-small-v1 | 96 | rf_split_temporal | 0.5419 | 0.6510 | 0.6667 | 0.0000 | 0.8000 | 0/6 |
| exp03-small-v1 | 96 | guided_fusion | 0.6374 | 0.5923 | 0.6667 | 0.4444 | 0.8000 | 0/6 |
| exp03-small-v1 | 96 | guided_no_local | 0.5481 | 0.5201 | 0.6667 | 0.5714 | 0.8000 | 0/6 |
| exp03-small-v1 | 96 | guided_no_scores | 0.5419 | 0.4964 | 0.4000 | 0.6667 | 0.0000 | 0/6 |
| exp03-small-v1 | 96 | anchored_fusion | 0.5750 | 0.5201 | 0.6667 | 0.0000 | 1.0000 | 0/6 |
| exp03-small-v1 | 96 | boundary_fusion | 0.6347 | 0.5923 | 0.6667 | 0.0000 | 1.0000 | 0/6 |
| exp03-small-v1 | 96 | boundary_no_local | 0.5509 | 0.5201 | 0.6667 | 0.0000 | 1.0000 | 0/6 |
| exp03-small-v1 | 96 | boundary_no_scores | 0.6423 | 0.6100 | 0.5000 | 0.0000 | 0.0000 | 0/6 |
| exp03-small-v1 | 96 | boundary_numeric_points | 0.6388 | 0.5923 | 0.6667 | 0.0000 | 0.8000 | 0/6 |

## 成对差值

形变日micro-F1，种子内站点bootstrap 2000次；条件于已拟合模型。

- boundary_fusion − anchored_fusion：+0.0722，95%CI [0.0000, 0.0871]。
- boundary_fusion − guided_fusion：+0.0000，95%CI [0.0000, 0.0000]。
- boundary_fusion − rf_split_temporal：-0.0587，95%CI [-0.6990, 0.3876]。
- boundary_fusion − boundary_no_local：+0.0722，95%CI [0.0000, 0.0871]。
- boundary_fusion − boundary_no_scores：-0.0176，95%CI [-0.1980, 0.0840]。
- boundary_fusion − boundary_numeric_points：+0.0000，95%CI [0.0000, 0.0000]。

全部四类型、±3日辅助及正常误报见CSV；缺测排除，无point adjustment。
开发验证不能充当未见测试；同分布仿真不能证明现场效果。完整论文复现或未运行方法不填成绩。
