# EXP15：局部跳变向量可视化核验（VEC-VLM）

种子2026101503，test，24同组站，模型qwen3-vl-plus。

| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 |
|---|---:|---:|---:|
| vector_visual | 0.7831 | 0.7779 | 0.9575 |
| vector_text | 0.8426 | 0.8404 | 0.9704 |
| vector_no_evidence | 0.7146 | 0.7231 | 0.9490 |
| vector_nearest | 0.8229 | 0.8212 | 0.9704 |
| pair_focus_repair | 0.7129 | 0.7105 | 0.9490 |
| segment_numeric | 0.7146 | 0.7231 | 0.9490 |
| rf_split_temporal | 0.4355 | 0.4731 | 0.5558 |
| rf_balanced | 0.4542 | 0.2412 | 0.6001 |

主候选比同模型原SDR+0.0702，比数值+0.0685；开发条件False。
旧控制请求与本轮新增请求分列于registration，不将复用控制算作新调用。
