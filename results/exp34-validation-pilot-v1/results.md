# EXP34：视觉选区—去叙述证据判定（VDE-GNSS）

72 fresh development stations, pilot.

| Method | Transient Aff | Daily F1 | Deformation Aff | Range union Aff | Four-type F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extra_trees | 0.3506 | 0.1645 | 0.8724 | 0.8735 | 0.3523 | 118.0000 | 458.0000 | 741.0000 |
| isolation_forest | — | — | — | — | — | — | — | — |
| jcv_image | 0.9137 | 0.7799 | 0.9946 | 0.9959 | 0.8076 | 790.0000 | 377.0000 | 69.0000 |
| jcv_image_pure | 0.5966 | 0.5429 | 0.0000 | 0.4252 | 0.1357 | 471.0000 | 405.0000 | 388.0000 |
| jcv_text_final | 0.8320 | 0.9027 | 0.9946 | 0.9937 | 0.8449 | 742.0000 | 43.0000 | 117.0000 |
| jcv_text_final_pure | 0.3620 | 0.4151 | 0.0000 | 0.2362 | 0.1038 | 247.0000 | 84.0000 | 612.0000 |
| jcv_text_pipeline | 0.8318 | 0.9104 | 0.9946 | 0.9937 | 0.8593 | 742.0000 | 29.0000 | 117.0000 |
| jcv_text_pipeline_pure | 0.3586 | 0.4970 | 0.0000 | 0.2354 | 0.1243 | 293.0000 | 27.0000 | 566.0000 |
| random_forest | 0.5385 | 0.2178 | 0.8951 | 0.8650 | 0.4018 | 197.0000 | 753.0000 | 662.0000 |
| rf_balanced | 0.8845 | 0.2580 | 0.9198 | 0.9182 | 0.4348 | 476.0000 | 2355.0000 | 383.0000 |
| rf_split_temporal | 0.4361 | 0.2369 | 0.9403 | 0.8800 | 0.4451 | 420.0000 | 2267.0000 | 439.0000 |
| segment_numeric | 0.5417 | 0.7460 | 0.9946 | 0.9927 | 0.7963 | 511.0000 | 0.0000 | 348.0000 |
| vector_nearest | 0.7916 | 0.9115 | 0.9946 | 0.9937 | 0.8672 | 721.0000 | 2.0000 | 138.0000 |

Mechanism confirmed: False
Stable system admitted: False

Pure outputs transient only and still receives numerical soft evidence.
Affiliation averages truth-positive stations; read negative-station FP alongside it.
