# GNSS-VLM-EXP05：GNSS视觉形态确认与数值精确定位实验

从所列运行的全站计数汇总；失败按零检测保留，无point adjustment。

| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 阶跃严格F1 | 失败 |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.3348 | 0.1509 | 0.6363 | 0.0000 | 0 |
| rf_balanced | 0.4426 | 0.2383 | 0.5565 | 0.1818 | 0 |
| extra_trees | 0.3049 | 0.1253 | 0.5790 | 0.0000 | 0 |
| isolation_forest | — | — | 0.4209 | — | 0 |
| rf_split_temporal | 0.4294 | 0.4057 | 0.5904 | 0.1053 | 0 |
| visual_only | 0.2980 | 0.2943 | 0.4801 | 0.1379 | 6 |
| guided_fusion | 0.5460 | 0.5233 | 0.7454 | 0.5000 | 0 |
| guided_no_local | 0.4742 | 0.4612 | 0.6398 | 0.4516 | 0 |
| guided_step_local | 0.4607 | 0.4587 | 0.5674 | 0.4000 | 1 |
| morphology_fusion | 0.6069 | 0.6001 | 0.7493 | 0.5185 | 0 |
| morphology_fixed_activity | 0.6076 | 0.5801 | 0.7479 | 0.5185 | 0 |
| morphology_no_local | 0.5785 | 0.5765 | 0.6853 | 0.5385 | 0 |
| morphology_global_fixed | 0.5832 | 0.5569 | 0.7042 | 0.5385 | 0 |
| morphology_step_local | 0.5651 | 0.5469 | 0.7094 | 0.4615 | 1 |
| morphology_step_fixed | 0.5589 | 0.5277 | 0.6807 | 0.4615 | 1 |
| localized_fusion | 0.6201 | 0.6134 | 0.7496 | 0.5714 | 0 |
| localized_fixed | 0.6208 | 0.5933 | 0.7482 | 0.5714 | 0 |
| localized_global | 0.5890 | 0.5871 | 0.6862 | 0.5806 | 0 |
| localized_global_fixed | 0.5937 | 0.5675 | 0.7049 | 0.5806 | 0 |
| localized_step_local | 0.5704 | 0.5522 | 0.7105 | 0.4828 | 1 |
| localized_step_fixed | 0.5642 | 0.5330 | 0.6813 | 0.4828 | 1 |
