# EXP16：候选返回对反向叠合视觉判别（PSC-VLM）

种子2026101508，validation，48同组站，模型qwen3-vl-plus。

| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 |
|---|---:|---:|---:|
| pair_overlap | 0.7866 | 0.7362 | 0.8646 |
| pair_text | 0.8123 | 0.7725 | 0.8725 |
| no_evidence | 0.7644 | 0.7341 | 0.8600 |
| vector_nearest | 0.8360 | 0.7836 | 0.8725 |
| pair_focus_repair | 0.7942 | 0.7462 | 0.8600 |
| segment_numeric | 0.7644 | 0.7341 | 0.8600 |
| rf_split_temporal | 0.4195 | 0.4112 | 0.6566 |
| rf_balanced | 0.4161 | 0.1985 | 0.6201 |

主候选比同模型原SDR-0.0076，比数值+0.0221；开发条件False。
旧控制请求与本轮新增请求分列于registration，不将复用控制算作新调用。
