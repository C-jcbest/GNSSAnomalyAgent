# EXP29-S3：视觉范围互补的独立确认

360 independent test stations.

| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |
|---|---:|---:|---:|---:|
| extra_trees | 0.4730 | 0.2067 | 0.3270 | 3444.0000 |
| isolation_forest | — | — | — | — |
| joint_vl_image | 0.9277 | 0.9207 | 0.8456 | 263.0000 |
| joint_vl_text | 0.9253 | 0.9219 | 0.8540 | 251.0000 |
| random_forest | 0.4967 | 0.1973 | 0.3476 | 5424.0000 |
| rf_balanced | 0.8716 | 0.2782 | 0.4249 | 10004.0000 |
| rf_split_temporal | 0.5233 | 0.2554 | 0.4267 | 9286.0000 |
| segment_numeric | 0.7742 | 0.8502 | 0.8206 | 37.0000 |
| vector_nearest | 0.9254 | 0.9246 | 0.8559 | 227.0000 |
| vf_vl_image_complement | 0.9254 | 0.9246 | 0.8559 | 227.0000 |
| vf_vl_image_pure | 0.3309 | 0.2847 | 0.0712 | 295.0000 |
| vf_vl_image_raw_complement | 0.9514 | 0.9089 | 0.8431 | 381.0000 |
| vf_vl_text_complement | 0.9254 | 0.9246 | 0.8559 | 227.0000 |
| vf_vl_text_pure | 0.0333 | 0.0027 | 0.0007 | 205.0000 |
| vf_vl_text_raw_complement | 0.9236 | 0.9021 | 0.8503 | 432.0000 |

Confirmation: False
