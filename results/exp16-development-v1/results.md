# EXP16：候选返回对反向叠合视觉判别（PSC-VLM）

种子2026101505，validation，24同组站，模型qwen3-vl-plus。

| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 |
|---|---:|---:|---:|
| pair_overlap | 0.8736 | 0.8482 | 0.9500 |
| pair_text | 0.8736 | 0.8482 | 0.9500 |
| no_evidence | 0.8736 | 0.8482 | 0.9500 |
| vector_nearest | 0.8736 | 0.8482 | 0.9500 |
| pair_focus_repair | 0.8736 | 0.8482 | 0.9500 |
| segment_numeric | 0.8736 | 0.8482 | 0.9500 |
| rf_split_temporal | 0.4812 | 0.4911 | 0.5326 |
| rf_balanced | 0.4154 | 0.2524 | 0.5290 |

主候选比同模型原SDR+0.0000，比数值+0.0000；开发条件False。
旧控制请求与本轮新增请求分列于registration，不将复用控制算作新调用。
