# EXP29：视觉先行定位与数值验证

validation, pilot, 60 stations.

| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |
|---|---:|---:|---:|---:|
| extra_trees | 0.6816 | 0.6007 | 0.4201 | 42.0000 |
| isolation_forest | — | — | — | — |
| nf_vl | 0.8883 | 0.9414 | 0.8312 | 0.0000 |
| numeric_all_pairs | 0.9433 | 0.9765 | 0.8516 | 0.0000 |
| random_forest | 0.7082 | 0.5316 | 0.4228 | 109.0000 |
| rf_balanced | 0.9502 | 0.4860 | 0.4403 | 659.0000 |
| rf_split_temporal | 0.6130 | 0.5638 | 0.4626 | 269.0000 |
| segment_numeric | 0.7778 | 0.8914 | 0.8111 | 0.0000 |
| vector_nearest | 0.9433 | 0.9765 | 0.8516 | 0.0000 |
| vf_flash_image_complement | 0.9433 | 0.9765 | 0.8516 | 0.0000 |
| vf_flash_image_pure | 0.9563 | 0.5780 | 0.1445 | 712.0000 |
| vf_flash_image_raw_complement | 0.9978 | 0.6323 | 0.7718 | 661.0000 |
| vf_flash_image_snap | 0.9712 | 0.8114 | 0.7896 | 204.0000 |
| vf_flash_image_verified | 0.6645 | 0.8285 | 0.8146 | 38.0000 |
| vf_vl_image_complement | 0.9433 | 0.9765 | 0.8516 | 0.0000 |
| vf_vl_image_pure | 0.4138 | 0.5062 | 0.1265 | 18.0000 |
| vf_vl_image_raw_complement | 0.9424 | 0.9697 | 0.8415 | 8.0000 |
| vf_vl_image_snap | 0.3629 | 0.5319 | 0.7404 | 23.0000 |
| vf_vl_image_verified | 0.2778 | 0.5233 | 0.7383 | 0.0000 |
| vf_vl_text_complement | 0.9433 | 0.9765 | 0.8516 | 0.0000 |
| vf_vl_text_pure | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| vf_vl_text_raw_complement | 0.9433 | 0.9765 | 0.8516 | 0.0000 |
| vf_vl_text_snap | 0.0000 | 0.0000 | 0.6075 | 0.0000 |
| vf_vl_text_verified | 0.0000 | 0.0000 | 0.6075 | 0.0000 |

Admission: False
