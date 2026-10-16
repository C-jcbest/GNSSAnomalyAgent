# EXP34-S1：去叙述联合视觉终判确认

36 fresh development stations, pilot.

| Method | Transient Aff | Daily F1 | Deformation Aff | Range union Aff | Four-type F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extra_trees | 0.5170 | 0.0674 | 0.8950 | 0.9106 | 0.2921 | 29.0000 | 323.0000 | 479.0000 |
| isolation_forest | — | — | — | — | — | — | — | — |
| jcv_image | 0.9995 | 0.8148 | 0.9316 | 0.9730 | 0.8133 | 508.0000 | 231.0000 | 0.0000 |
| jcv_image_pure | 0.6523 | 0.5311 | 0.0000 | 0.4211 | 0.1328 | 269.0000 | 236.0000 | 239.0000 |
| jcv_text_final | 1.0000 | 0.9932 | 0.9316 | 0.9783 | 0.8302 | 508.0000 | 7.0000 | 0.0000 |
| jcv_text_final_pure | 0.5000 | 0.6518 | 0.0000 | 0.3139 | 0.1630 | 249.0000 | 7.0000 | 259.0000 |
| jcv_text_pipeline | 0.9652 | 0.8545 | 0.9316 | 0.9738 | 0.7851 | 508.0000 | 173.0000 | 0.0000 |
| jcv_text_pipeline_pure | 0.3670 | 0.3058 | 0.0000 | 0.2360 | 0.0764 | 124.0000 | 179.0000 | 384.0000 |
| random_forest | 0.5612 | 0.0914 | 0.8819 | 0.9060 | 0.3324 | 43.0000 | 390.0000 | 465.0000 |
| rf_balanced | 0.9469 | 0.2577 | 0.9099 | 0.9123 | 0.4539 | 229.0000 | 1040.0000 | 279.0000 |
| rf_split_temporal | 0.3857 | 0.1221 | 0.9024 | 0.9148 | 0.4041 | 98.0000 | 999.0000 | 410.0000 |
| segment_numeric | 0.6667 | 0.7700 | 0.9316 | 0.9378 | 0.7373 | 318.0000 | 0.0000 | 190.0000 |
| vector_nearest | 1.0000 | 1.0000 | 0.9316 | 0.9783 | 0.8494 | 508.0000 | 0.0000 | 0.0000 |

Mechanism confirmed: False
Stable system admitted: False

Pure outputs transient only and still receives numerical soft evidence.
Affiliation averages truth-positive stations; read negative-station FP alongside it.
