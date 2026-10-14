# GNSS-VLM-EXP05：GNSS视觉形态确认与数值精确定位实验

从所列运行的全站计数汇总；失败按零检测保留，无point adjustment。

| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 阶跃严格F1 | 失败 |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.3808 | 0.2292 | 0.8038 | 0.0000 | 0 |
| rf_balanced | 0.6551 | 0.4596 | 0.7215 | 0.6667 | 0 |
| extra_trees | 0.2272 | 0.0333 | 0.7143 | 0.0000 | 0 |
| isolation_forest | — | — | 0.4810 | — | 0 |
| rf_split_temporal | 0.6723 | 0.6167 | 0.7554 | 0.6667 | 0 |
| guided_fusion | 0.4365 | 0.4250 | 0.8398 | 0.4000 | 0 |
| guided_no_local | 0.4850 | 0.5167 | 0.8622 | 0.6667 | 0 |
| guided_step_local | 0.4649 | 0.5000 | 0.7788 | 0.5000 | 0 |
| morphology_fusion | 0.5615 | 0.5500 | 0.8472 | 0.4000 | 0 |
| morphology_fixed_activity | 0.5358 | 0.5500 | 0.7524 | 0.4000 | 0 |
| morphology_no_local | 0.6350 | 0.6667 | 0.8739 | 0.6667 | 0 |
| morphology_global_fixed | 0.6107 | 0.6667 | 0.7846 | 0.6667 | 0 |
| morphology_step_local | 0.5899 | 0.6250 | 0.7864 | 0.5000 | 0 |
| morphology_step_fixed | 0.5690 | 0.6250 | 0.7846 | 0.5000 | 0 |
| localized_fusion | 0.5615 | 0.5500 | 0.8515 | 0.4000 | 0 |
| localized_fixed | 0.5358 | 0.5500 | 0.7563 | 0.4000 | 0 |
| localized_global | 0.6350 | 0.6667 | 0.8739 | 0.6667 | 0 |
| localized_global_fixed | 0.6107 | 0.6667 | 0.7846 | 0.6667 | 0 |
| localized_step_local | 0.5899 | 0.6250 | 0.7864 | 0.5000 | 0 |
| localized_step_fixed | 0.5690 | 0.6250 | 0.7846 | 0.5000 | 0 |
