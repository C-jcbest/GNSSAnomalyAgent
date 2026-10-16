# EXP32：联合上下文视觉识别确认（JVC-GNSS）

72 development stations, pilot.

| Method | Transient Aff | Daily F1 | Deformation Aff | Range union Aff | Four-type F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extra_trees | 0.5272 | 0.2667 | 0.8403 | 0.8765 | 0.3453 | 232.0000 | 562.0000 | 714.0000 |
| isolation_forest | — | — | — | — | — | — | — | — |
| jcv_global_only | 0.7367 | 0.7786 | 0.9105 | 0.9073 | 0.7963 | 668.0000 | 102.0000 | 278.0000 |
| jcv_global_only_pure | 0.1528 | 0.1117 | 0.0000 | 0.0978 | 0.0279 | 62.0000 | 102.0000 | 884.0000 |
| jcv_image | 0.7493 | 0.6955 | 0.9105 | 0.9052 | 0.7700 | 668.0000 | 307.0000 | 278.0000 |
| jcv_image_no_numeric | 0.7514 | 0.7669 | 0.9105 | 0.9061 | 0.8037 | 668.0000 | 128.0000 | 278.0000 |
| jcv_image_no_numeric_pure | 0.1508 | 0.0725 | 0.0000 | 0.0966 | 0.0181 | 42.0000 | 171.0000 | 904.0000 |
| jcv_image_pure | 0.2515 | 0.1057 | 0.0000 | 0.1471 | 0.0264 | 71.0000 | 327.0000 | 875.0000 |
| jcv_local_only | 0.7231 | 0.7995 | 0.9105 | 0.9010 | 0.8068 | 668.0000 | 57.0000 | 278.0000 |
| jcv_local_only_pure | 0.1757 | 0.1797 | 0.0000 | 0.1084 | 0.0449 | 101.0000 | 77.0000 | 845.0000 |
| jcv_text_final | 0.7083 | 0.7827 | 0.9105 | 0.9012 | 0.7973 | 668.0000 | 93.0000 | 278.0000 |
| jcv_text_final_pure | 0.1636 | 0.1921 | 0.0000 | 0.1328 | 0.0480 | 115.0000 | 136.0000 | 831.0000 |
| jcv_text_pipeline | 0.7226 | 0.7714 | 0.9105 | 0.9047 | 0.8048 | 668.0000 | 118.0000 | 278.0000 |
| jcv_text_pipeline_pure | 0.1207 | 0.1002 | 0.0000 | 0.0871 | 0.0251 | 64.0000 | 267.0000 | 882.0000 |
| random_forest | 0.5792 | 0.3154 | 0.8344 | 0.8813 | 0.3737 | 358.0000 | 966.0000 | 588.0000 |
| rf_balanced | 0.8237 | 0.2805 | 0.8944 | 0.9097 | 0.4054 | 534.0000 | 2327.0000 | 412.0000 |
| rf_split_temporal | 0.4492 | 0.2688 | 0.8668 | 0.8846 | 0.4144 | 470.0000 | 2081.0000 | 476.0000 |
| segment_numeric | 0.7083 | 0.8278 | 0.9105 | 0.9022 | 0.8189 | 668.0000 | 0.0000 | 278.0000 |
| vector_nearest | 0.7083 | 0.8278 | 0.9105 | 0.9022 | 0.8189 | 668.0000 | 0.0000 | 278.0000 |

Mechanism confirmed: False
Stable system admitted: False

Pure outputs transient only; their four-type F1 is not a full-system ranking.
Pure still receives numerical soft evidence except no_numeric.
Affiliation averages truth-positive stations; negative-station false alarms are separate.
