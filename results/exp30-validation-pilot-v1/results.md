# EXP30：视觉粗筛与自适应局部重定界（AIR-GNSS）

72 stations, validation, pilot.

| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |
|---|---:|---:|---:|---:|
| air_global | 0.9486 | 0.8992 | 0.8417 | 83.0000 |
| air_global_pure | 0.1615 | 0.0182 | 0.0046 | 62.0000 |
| air_image | 0.9093 | 0.8747 | 0.8418 | 118.0000 |
| air_image_no_numeric | 0.9081 | 0.8678 | 0.8401 | 131.0000 |
| air_image_no_numeric_pure | 0.1266 | 0.0674 | 0.0168 | 111.0000 |
| air_image_pure | 0.1284 | 0.0914 | 0.0228 | 112.0000 |
| air_text_final | 0.9084 | 0.8519 | 0.8361 | 162.0000 |
| air_text_final_pure | 0.1253 | 0.0530 | 0.0133 | 148.0000 |
| air_text_pipeline | 0.9046 | 0.8647 | 0.8393 | 137.0000 |
| air_text_pipeline_pure | 0.0617 | 0.0000 | 0.0000 | 116.0000 |
| extra_trees | 0.6691 | 0.4005 | 0.3739 | 459.0000 |
| isolation_forest | — | — | — | — |
| random_forest | 0.6709 | 0.3448 | 0.4032 | 974.0000 |
| rf_balanced | 0.8980 | 0.3192 | 0.4479 | 2073.0000 |
| rf_split_temporal | 0.6657 | 0.3190 | 0.4659 | 2018.0000 |
| segment_numeric | 0.6667 | 0.8009 | 0.7971 | 0.0000 |
| vector_nearest | 0.9103 | 0.9295 | 0.8555 | 21.0000 |
| vrc_image | 0.9051 | 0.8899 | 0.8456 | 90.0000 |
| vrc_text | 0.9068 | 0.9171 | 0.8524 | 42.0000 |

Expansion/adoption gate: False

Pure methods output transient only; four-type F1 is not a fair full-system comparison.
