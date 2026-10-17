# GNSS-VLM-EXP36：分类型统计

原批次18站；读取冻结结果，新增模型请求0。每个异常类型独立统计，不报告跨类型平均。

## 各类型主指标

| 方法 | 输出范围 | 单点日F1 | 阶跃日F1 | 暂态Aff-F1 | 形变Aff-F1 |
|---|---|---:|---:|---:|---:|
| rf_split_temporal | 四类输出 | 1.0000 | 0.0000 | 0.6165 | 0.8462 |
| segment_numeric | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| random_forest | 四类输出 | 0.5000 | 0.0000 | 0.2836 | 0.8152 |
| rf_balanced | 四类输出 | 0.9091 | 0.0000 | 0.7624 | 0.8825 |
| extra_trees | 四类输出 | 0.5000 | 0.0000 | 0.1447 | 0.8227 |
| isolation_forest | 仅通用异常 | — | — | — | — |
| vector_nearest | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_image_raw | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_image_raw_pure | 仅暂态输出 | — | — | 0.3333 | — |
| cev_image_refined | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_image_refined_pure | 仅暂态输出 | — | — | 0.3333 | — |
| cev_image_verified | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_image_verified_pure | 仅暂态输出 | — | — | 0.3333 | — |
| cev_no_examples_raw | 四类输出 | 1.0000 | 0.8333 | 0.8048 | 0.9172 |
| cev_no_examples_raw_pure | 仅暂态输出 | — | — | 0.4717 | — |
| cev_no_examples_refined | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_no_examples_refined_pure | 仅暂态输出 | — | — | 0.3333 | — |
| cev_no_examples_verified | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_no_examples_verified_pure | 仅暂态输出 | — | — | 0.3333 | — |
| cev_text_raw | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_text_raw_pure | 仅暂态输出 | — | — | 0.3333 | — |
| cev_text_refined | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_text_refined_pure | 仅暂态输出 | — | — | 0.3333 | — |
| cev_text_verified | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| cev_text_verified_pure | 仅暂态输出 | — | — | 0.3333 | — |
| gnv_image | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |
| numeric_all_pairs | 四类输出 | 1.0000 | 0.8333 | 0.6664 | 0.9172 |

## 单点异常

主评价：日级统计。

| 方法 | 日Precision | 日Recall | 日F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rf_split_temporal | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| segment_numeric | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| random_forest | 1.0000 | 0.3333 | 0.5000 | 2 | 0 | 4 |
| rf_balanced | 1.0000 | 0.8333 | 0.9091 | 5 | 0 | 1 |
| extra_trees | 1.0000 | 0.3333 | 0.5000 | 2 | 0 | 4 |
| isolation_forest | — | — | — | — | — | — |
| vector_nearest | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_image_raw | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_image_raw_pure | — | — | — | — | — | — |
| cev_image_refined | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_image_refined_pure | — | — | — | — | — | — |
| cev_image_verified | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_image_verified_pure | — | — | — | — | — | — |
| cev_no_examples_raw | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_no_examples_raw_pure | — | — | — | — | — | — |
| cev_no_examples_refined | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_no_examples_refined_pure | — | — | — | — | — | — |
| cev_no_examples_verified | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_no_examples_verified_pure | — | — | — | — | — | — |
| cev_text_raw | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_text_raw_pure | — | — | — | — | — | — |
| cev_text_refined | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_text_refined_pure | — | — | — | — | — | — |
| cev_text_verified | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| cev_text_verified_pure | — | — | — | — | — | — |
| gnv_image | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |
| numeric_all_pairs | 1.0000 | 1.0000 | 1.0000 | 6 | 0 | 0 |

## 永久阶跃

主评价：日级统计。

| 方法 | 日Precision | 日Recall | 日F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rf_split_temporal | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 6 |
| segment_numeric | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| random_forest | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 6 |
| rf_balanced | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 6 |
| extra_trees | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 6 |
| isolation_forest | — | — | — | — | — | — |
| vector_nearest | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_image_raw | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_image_raw_pure | — | — | — | — | — | — |
| cev_image_refined | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_image_refined_pure | — | — | — | — | — | — |
| cev_image_verified | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_image_verified_pure | — | — | — | — | — | — |
| cev_no_examples_raw | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_no_examples_raw_pure | — | — | — | — | — | — |
| cev_no_examples_refined | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_no_examples_refined_pure | — | — | — | — | — | — |
| cev_no_examples_verified | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_no_examples_verified_pure | — | — | — | — | — | — |
| cev_text_raw | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_text_raw_pure | — | — | — | — | — | — |
| cev_text_refined | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_text_refined_pure | — | — | — | — | — | — |
| cev_text_verified | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| cev_text_verified_pure | — | — | — | — | — | — |
| gnv_image | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |
| numeric_all_pairs | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 |

## 暂态异常

| 方法 | Aff-P | Aff-R | Aff-F1 | 该类阳性站 | 无该类异常站FP日 |
|---|---:|---:|---:|---:|---:|
| rf_split_temporal | 0.5901 | 0.6509 | 0.6165 | 6 | 146 |
| segment_numeric | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| random_forest | 0.2631 | 0.3186 | 0.2836 | 6 | 139 |
| rf_balanced | 0.7222 | 0.8195 | 0.7624 | 6 | 224 |
| extra_trees | 0.1475 | 0.1419 | 0.1447 | 6 | 149 |
| isolation_forest | — | — | — | — | — |
| vector_nearest | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_image_raw | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_image_raw_pure | 0.3333 | 0.3332 | 0.3333 | 6 | 0 |
| cev_image_refined | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_image_refined_pure | 0.3333 | 0.3333 | 0.3333 | 6 | 0 |
| cev_image_verified | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_image_verified_pure | 0.3333 | 0.3333 | 0.3333 | 6 | 0 |
| cev_no_examples_raw | 0.8032 | 0.8064 | 0.8048 | 6 | 14 |
| cev_no_examples_raw_pure | 0.4703 | 0.4731 | 0.4717 | 6 | 14 |
| cev_no_examples_refined | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_no_examples_refined_pure | 0.3333 | 0.3333 | 0.3333 | 6 | 0 |
| cev_no_examples_verified | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_no_examples_verified_pure | 0.3333 | 0.3333 | 0.3333 | 6 | 0 |
| cev_text_raw | 0.6662 | 0.6667 | 0.6664 | 6 | 6 |
| cev_text_raw_pure | 0.3333 | 0.3333 | 0.3333 | 6 | 6 |
| cev_text_refined | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_text_refined_pure | 0.3333 | 0.3333 | 0.3333 | 6 | 0 |
| cev_text_verified | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |
| cev_text_verified_pure | 0.3333 | 0.3333 | 0.3333 | 6 | 0 |
| gnv_image | 0.6662 | 0.6667 | 0.6664 | 6 | 125 |
| numeric_all_pairs | 0.6662 | 0.6667 | 0.6664 | 6 | 0 |

辅助日级统计：

| 方法 | 日Precision | 日Recall | 日F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rf_split_temporal | 0.2756 | 0.5440 | 0.3659 | 105 | 276 | 88 |
| segment_numeric | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| random_forest | 0.0655 | 0.0570 | 0.0609 | 11 | 157 | 182 |
| rf_balanced | 0.1971 | 0.4870 | 0.2806 | 94 | 383 | 99 |
| extra_trees | 0.0066 | 0.0052 | 0.0058 | 1 | 151 | 192 |
| isolation_forest | — | — | — | — | — | — |
| vector_nearest | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_image_raw | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_image_raw_pure | 1.0000 | 0.2694 | 0.4245 | 52 | 0 | 141 |
| cev_image_refined | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_image_refined_pure | 1.0000 | 0.2902 | 0.4498 | 56 | 0 | 137 |
| cev_image_verified | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_image_verified_pure | 1.0000 | 0.2902 | 0.4498 | 56 | 0 | 137 |
| cev_no_examples_raw | 0.8636 | 0.6891 | 0.7666 | 133 | 21 | 60 |
| cev_no_examples_raw_pure | 0.7015 | 0.2435 | 0.3615 | 47 | 20 | 146 |
| cev_no_examples_refined | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_no_examples_refined_pure | 1.0000 | 0.2435 | 0.3917 | 47 | 0 | 146 |
| cev_no_examples_verified | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_no_examples_verified_pure | 1.0000 | 0.2435 | 0.3917 | 47 | 0 | 146 |
| cev_text_raw | 0.9500 | 0.6891 | 0.7988 | 133 | 7 | 60 |
| cev_text_raw_pure | 0.9032 | 0.2902 | 0.4392 | 56 | 6 | 137 |
| cev_text_refined | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_text_refined_pure | 1.0000 | 0.2902 | 0.4498 | 56 | 0 | 137 |
| cev_text_verified | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |
| cev_text_verified_pure | 1.0000 | 0.2902 | 0.4498 | 56 | 0 | 137 |
| gnv_image | 0.5135 | 0.6891 | 0.5885 | 133 | 126 | 60 |
| numeric_all_pairs | 0.9925 | 0.6891 | 0.8135 | 133 | 1 | 60 |

## 持续形变

| 方法 | Aff-P | Aff-R | Aff-F1 | 该类阳性站 | 无该类异常站FP日 |
|---|---:|---:|---:|---:|---:|
| rf_split_temporal | 0.9300 | 0.8092 | 0.8462 | 6 | 109 |
| segment_numeric | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| random_forest | 0.9405 | 0.7523 | 0.8152 | 6 | 40 |
| rf_balanced | 0.8391 | 0.9345 | 0.8825 | 6 | 230 |
| extra_trees | 0.9689 | 0.7420 | 0.8227 | 6 | 52 |
| isolation_forest | — | — | — | — | — |
| vector_nearest | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_image_raw | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_image_raw_pure | — | — | — | — | — |
| cev_image_refined | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_image_refined_pure | — | — | — | — | — |
| cev_image_verified | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_image_verified_pure | — | — | — | — | — |
| cev_no_examples_raw | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_no_examples_raw_pure | — | — | — | — | — |
| cev_no_examples_refined | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_no_examples_refined_pure | — | — | — | — | — |
| cev_no_examples_verified | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_no_examples_verified_pure | — | — | — | — | — |
| cev_text_raw | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_text_raw_pure | — | — | — | — | — |
| cev_text_refined | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_text_refined_pure | — | — | — | — | — |
| cev_text_verified | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| cev_text_verified_pure | — | — | — | — | — |
| gnv_image | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |
| numeric_all_pairs | 0.9924 | 0.8748 | 0.9172 | 6 | 0 |

辅助日级统计：

| 方法 | 日Precision | 日Recall | 日F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rf_split_temporal | 0.6351 | 0.7921 | 0.7050 | 564 | 324 | 148 |
| segment_numeric | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| random_forest | 0.8778 | 0.6053 | 0.7165 | 431 | 60 | 281 |
| rf_balanced | 0.5865 | 0.7907 | 0.6734 | 563 | 397 | 149 |
| extra_trees | 0.8747 | 0.5590 | 0.6821 | 398 | 57 | 314 |
| isolation_forest | — | — | — | — | — | — |
| vector_nearest | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_image_raw | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_image_raw_pure | — | — | — | — | — | — |
| cev_image_refined | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_image_refined_pure | — | — | — | — | — | — |
| cev_image_verified | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_image_verified_pure | — | — | — | — | — | — |
| cev_no_examples_raw | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_no_examples_raw_pure | — | — | — | — | — | — |
| cev_no_examples_refined | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_no_examples_refined_pure | — | — | — | — | — | — |
| cev_no_examples_verified | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_no_examples_verified_pure | — | — | — | — | — | — |
| cev_text_raw | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_text_raw_pure | — | — | — | — | — | — |
| cev_text_refined | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_text_refined_pure | — | — | — | — | — | — |
| cev_text_verified | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| cev_text_verified_pure | — | — | — | — | — | — |
| gnv_image | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |
| numeric_all_pairs | 0.9703 | 0.9621 | 0.9661 | 685 | 21 | 27 |

“—”表示不适用。仅暂态输出的方法不评价其他三类，通用检测器不编造类型结果。
Aff只对有该类真值的站聚合，需与该类全站日级指标及负站误报并读。
形变沿用数值输出时不能将其成绩归为新增视觉作用；原冻结准入决定不因展示修正改变。
