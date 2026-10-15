# EXP15：局部跳变向量可视化核验（VEC-VLM）

种子2026101501，validation，24同组站，模型qwen3-vl-plus。

| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 |
|---|---:|---:|---:|
| vector_visual | 0.8393 | 0.8211 | 0.9195 |
| vector_text | 0.8098 | 0.7947 | 0.9147 |
| vector_no_evidence | 0.7544 | 0.7434 | 0.9147 |
| vector_nearest | 0.8393 | 0.8211 | 0.9195 |
| pair_focus_repair | 0.7544 | 0.7434 | 0.9147 |
| segment_numeric | 0.7544 | 0.7434 | 0.9147 |
| rf_split_temporal | 0.4423 | 0.4076 | 0.5654 |
| rf_balanced | 0.4283 | 0.2230 | 0.5637 |

主候选比同模型原SDR+0.0849，比数值+0.0849；开发条件True。
旧控制请求与本轮新增请求分列于registration，不将复用控制算作新调用。
