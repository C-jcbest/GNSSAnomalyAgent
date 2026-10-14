# GNSS-VLM-EXP05：GNSS视觉形态确认与数值精确定位实验

从所列运行的全站计数汇总；失败按零检测保留，无point adjustment。

| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 阶跃严格F1 | 失败 |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.3091 | 0.1092 | 0.6376 | 0.0000 | 0 |
| rf_balanced | 0.4316 | 0.2582 | 0.5574 | 0.1818 | 0 |
| extra_trees | 0.2727 | 0.0678 | 0.6215 | 0.0000 | 0 |
| isolation_forest | — | — | 0.4490 | — | 0 |
| rf_split_temporal | 0.4556 | 0.4679 | 0.5944 | 0.1667 | 0 |
| guided_fusion | 0.4986 | 0.5271 | 0.8028 | 0.3333 | 0 |
| guided_no_local | 0.5356 | 0.5668 | 0.7264 | 0.4286 | 0 |
| guided_step_local | 0.4091 | 0.4427 | 0.8231 | 0.3333 | 1 |
| morphology_fusion | 0.6153 | 0.6438 | 0.8084 | 0.4000 | 0 |
| morphology_fixed_activity | 0.5775 | 0.5813 | 0.6992 | 0.4000 | 0 |
| morphology_no_local | 0.6692 | 0.7048 | 0.7563 | 0.5000 | 0 |
| morphology_global_fixed | 0.6514 | 0.6736 | 0.6972 | 0.5000 | 0 |
| morphology_step_local | 0.5106 | 0.5441 | 0.8297 | 0.3529 | 1 |
| morphology_step_fixed | 0.4639 | 0.4853 | 0.6949 | 0.3529 | 1 |
| localized_fusion | 0.6329 | 0.6614 | 0.8108 | 0.4706 | 0 |
| localized_fixed | 0.5952 | 0.5989 | 0.7012 | 0.4706 | 0 |
| localized_global | 0.6980 | 0.7337 | 0.7563 | 0.6154 | 0 |
| localized_global_fixed | 0.6803 | 0.7024 | 0.6972 | 0.6154 | 0 |
| localized_step_local | 0.5335 | 0.5670 | 0.8304 | 0.4444 | 1 |
| localized_step_fixed | 0.4867 | 0.5082 | 0.6955 | 0.4444 | 1 |
