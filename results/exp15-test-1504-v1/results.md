# EXP15：局部跳变向量可视化核验（VEC-VLM）

种子2026101504，test，24同组站，模型qwen3-vl-plus。

| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 |
|---|---:|---:|---:|
| vector_visual | 0.7690 | 0.7341 | 0.9070 |
| vector_text | 0.8298 | 0.7976 | 0.9148 |
| vector_no_evidence | 0.7817 | 0.7534 | 0.9070 |
| vector_nearest | 0.8298 | 0.7976 | 0.9148 |
| pair_focus_repair | 0.7817 | 0.7534 | 0.9070 |
| segment_numeric | 0.7817 | 0.7534 | 0.9070 |
| rf_split_temporal | 0.5197 | 0.5320 | 0.7087 |
| rf_balanced | 0.4397 | 0.2201 | 0.6015 |

主候选比同模型原SDR-0.0127，比数值-0.0127；开发条件False。
旧控制请求与本轮新增请求分列于registration，不将复用控制算作新调用。
