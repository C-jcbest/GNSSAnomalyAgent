# EXP30：视觉粗筛与自适应局部重定界（AIR-GNSS）

原连接故障运行：288请求全部ConnectError，0模型回答。下表为事先规定的原近邻回退评分，
不是AIR方法能力结果，也不用于否定研究假设。独立技术复测另报，原失败成本不删除。

72 stations, validation, pilot.

| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |
|---|---:|---:|---:|---:|
| air_global | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| air_global_pure | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| air_image | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| air_image_no_numeric | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| air_image_no_numeric_pure | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| air_image_pure | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| air_text_final | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| air_text_final_pure | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| air_text_pipeline | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| air_text_pipeline_pure | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| extra_trees | 0.6691 | 0.4005 | 0.3739 | 459.0000 |
| isolation_forest | — | — | — | — |
| random_forest | 0.6709 | 0.3448 | 0.4032 | 974.0000 |
| rf_balanced | 0.8980 | 0.3192 | 0.4479 | 2073.0000 |
| rf_split_temporal | 0.6657 | 0.3190 | 0.4659 | 2018.0000 |
| segment_numeric | 0.6667 | 0.8009 | 0.7971 | 0.0000 |
| vector_nearest | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| vrc_image | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| vrc_text | 0.9103 | 0.9295 | 0.8555 | 21.0000 |

Expansion/adoption gate: False

Pure methods output transient only; four-type F1 is not a fair full-system comparison.
