# GNSS-VLM-EXP06：GNSS位移结构分解与视觉语义核验实验

SDR-VLM retains independent numerical decisions when visual repair fails; parent visual failures separately disclosed. Visual-veto/visual-only failures use zero detections. Development stations were used for method selection; no point adjustment.

主方法SDR-VLM=`pair_focus_repair`。派生方法无新增请求，视觉成本按pair_focus父方法理解。

| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 尖峰 | 阶跃严格 | 暂态 | 活动 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_forest | 0.2723 | 0.1249 | 0.5012 | 0.4211 | 0.0000 | 0.1784 | 0.4898 |
| rf_balanced | 0.4330 | 0.2493 | 0.5112 | 0.7200 | 0.2400 | 0.3119 | 0.4602 |
| extra_trees | 0.2209 | 0.0951 | 0.3934 | 0.3333 | 0.0000 | 0.1583 | 0.3920 |
| isolation_forest | — | — | 0.3710 | — | — | — | — |
| rf_split_temporal | 0.4188 | 0.3715 | 0.5460 | 0.8000 | 0.0000 | 0.3755 | 0.4995 |
| segment_numeric | 0.8323 | 0.8224 | 0.9141 | 0.8000 | 0.7500 | 0.8788 | 0.9002 |
| visual_only | 0.1138 | 0.1337 | 0.3486 | 0.0000 | 0.0000 | 0.1120 | 0.3432 |
| guided_fusion | 0.4650 | 0.4299 | 0.5550 | 0.4615 | 0.3333 | 0.5958 | 0.4694 |
| morphology_fusion | 0.5672 | 0.5012 | 0.6304 | 0.8000 | 0.3704 | 0.5625 | 0.5359 |
| localized_fusion | 0.5817 | 0.5157 | 0.6302 | 0.8000 | 0.4286 | 0.5625 | 0.5359 |
| pair_focus（配对子任务输出） | — | — | — | — | 0.3158 | 0.3711 | — |
| pair_focus_repair | 0.8652 | 0.8523 | 0.9178 | 0.8000 | 0.8235 | 0.9372 | 0.9002 |
| pair_raw（配对子任务输出） | — | — | — | — | 0.3636 | 0.4611 | — |
| pair_raw_repair | 0.8622 | 0.8523 | 0.9215 | 0.8000 | 0.8235 | 0.9251 | 0.9002 |

父视觉失败：当前主方法输入0次，旧完整提示0次，纯视觉1次；全部保留，无重试。
