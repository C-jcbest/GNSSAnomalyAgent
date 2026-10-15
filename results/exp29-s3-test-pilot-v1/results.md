# EXP29-S3：视觉范围互补的独立确认

60 independent test stations.

| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |
|---|---:|---:|---:|---:|
| extra_trees | 0.3611 | 0.2003 | 0.3574 | 487.0000 |
| isolation_forest | — | — | — | — |
| joint_vl_image | 0.7633 | 0.7737 | 0.7508 | 71.0000 |
| joint_vl_text | 0.7633 | 0.7737 | 0.7508 | 71.0000 |
| random_forest | 0.4317 | 0.2158 | 0.3545 | 592.0000 |
| rf_balanced | 0.8787 | 0.3076 | 0.3780 | 1481.0000 |
| rf_split_temporal | 0.4347 | 0.2370 | 0.3758 | 1508.0000 |
| segment_numeric | 0.6666 | 0.8029 | 0.7641 | 1.0000 |
| vector_nearest | 0.7633 | 0.7737 | 0.7508 | 71.0000 |
| vf_vl_image_complement | 0.7633 | 0.7737 | 0.7508 | 71.0000 |
| vf_vl_image_pure | 0.2236 | 0.2046 | 0.0511 | 6.0000 |
| vf_vl_image_raw_complement | 0.8489 | 0.7746 | 0.7333 | 73.0000 |
| vf_vl_text_complement | 0.7633 | 0.7737 | 0.7508 | 71.0000 |
| vf_vl_text_pure | 0.0000 | 0.0000 | 0.0000 | 5.0000 |
| vf_vl_text_raw_complement | 0.7633 | 0.7701 | 0.7499 | 76.0000 |

Confirmation: False
