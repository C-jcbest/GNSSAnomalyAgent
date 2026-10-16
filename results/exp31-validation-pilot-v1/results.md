# EXP31：全局—局部联合视觉复核（JCV-GNSS）

36 stations, validation, pilot.

| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| air_global | 0.9961 | 0.9872 | 0.8697 | 423.0000 | 11.0000 | 0.0000 |
| air_global_pure | 0.1104 | 0.1085 | 0.0271 | 25.0000 | 13.0000 | 398.0000 |
| air_image | 1.0000 | 0.9442 | 0.8589 | 423.0000 | 50.0000 | 0.0000 |
| air_image_pure | 0.0831 | 0.1000 | 0.0250 | 25.0000 | 52.0000 | 398.0000 |
| extra_trees | 0.5689 | 0.3902 | 0.3336 | 136.0000 | 138.0000 | 287.0000 |
| isolation_forest | — | — | — | — | — | — |
| jcv_global_only | 1.0000 | 0.9614 | 0.8632 | 423.0000 | 34.0000 | 0.0000 |
| jcv_global_only_pure | 0.2911 | 0.3040 | 0.0760 | 83.0000 | 40.0000 | 340.0000 |
| jcv_image | 1.0000 | 0.9421 | 0.8584 | 423.0000 | 52.0000 | 0.0000 |
| jcv_image_no_numeric | 1.0000 | 0.9266 | 0.8545 | 423.0000 | 67.0000 | 0.0000 |
| jcv_image_no_numeric_pure | 0.0831 | 0.0893 | 0.0223 | 23.0000 | 69.0000 | 400.0000 |
| jcv_image_pure | 0.3744 | 0.3926 | 0.0982 | 117.0000 | 56.0000 | 306.0000 |
| jcv_local_only | 1.0000 | 0.9453 | 0.8592 | 423.0000 | 49.0000 | 0.0000 |
| jcv_local_only_pure | 0.1795 | 0.1741 | 0.0435 | 45.0000 | 49.0000 | 378.0000 |
| jcv_text_final | 1.0000 | 0.9442 | 0.8589 | 423.0000 | 50.0000 | 0.0000 |
| jcv_text_final_pure | 0.3177 | 0.2993 | 0.0748 | 85.0000 | 60.0000 | 338.0000 |
| jcv_text_pipeline | 0.9996 | 0.9872 | 0.8697 | 423.0000 | 11.0000 | 0.0000 |
| jcv_text_pipeline_pure | 0.0605 | 0.0000 | 0.0000 | 0.0000 | 11.0000 | 423.0000 |
| random_forest | 0.7025 | 0.4047 | 0.4109 | 138.0000 | 121.0000 | 285.0000 |
| rf_balanced | 0.8553 | 0.3474 | 0.4099 | 231.0000 | 676.0000 | 192.0000 |
| rf_split_temporal | 0.5235 | 0.3506 | 0.4358 | 196.0000 | 499.0000 | 227.0000 |
| segment_numeric | 0.8333 | 0.9358 | 0.8254 | 372.0000 | 0.0000 | 51.0000 |
| vector_nearest | 1.0000 | 1.0000 | 0.8729 | 423.0000 | 0.0000 | 0.0000 |
| vrc_image | 1.0000 | 0.9883 | 0.8699 | 423.0000 | 10.0000 | 0.0000 |
| vrc_text | 1.0000 | 0.9965 | 0.8720 | 423.0000 | 3.0000 | 0.0000 |

Preregistered decision: False

Pure methods output transient only; their four-type F1 is not a full-system comparison.
Affiliation averages truth-positive stations; read negative-station FP and daily scores alongside it.
