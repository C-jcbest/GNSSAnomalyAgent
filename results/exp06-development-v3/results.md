# GNSS-VLM-EXP06：GNSS位移结构分解与视觉语义核验实验

SDR-VLM retains independent numerical decisions when visual repair fails; parent visual failures separately disclosed. Visual-veto/visual-only failures use zero detections. Development stations were used for method selection; no point adjustment.

主方法SDR-VLM=`pair_focus_repair`。派生方法无新增请求，视觉成本按pair_focus父方法理解。

| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 尖峰 | 阶跃严格 | 暂态 | 活动 |
|---|---:|---:|---:|---:|---:|---:|---:|
| segment_numeric | 0.7953 | 0.7971 | 0.8935 | 0.8571 | 0.7619 | 0.6829 | 0.8791 |
| pair_focus | 0.0843 | 0.0833 | 0.2496 | 0.0000 | 0.0000 | 0.3374 | 0.0000 |
| pair_focus_repair | 0.8233 | 0.8018 | 0.9163 | 0.8571 | 0.8000 | 0.7570 | 0.8791 |
| random_forest | 0.4159 | 0.2183 | 0.6044 | 0.6667 | 0.0000 | 0.4382 | 0.5589 |
| rf_balanced | 0.5009 | 0.2818 | 0.5811 | 0.8000 | 0.2857 | 0.3804 | 0.5375 |
| rf_split_temporal | 0.5408 | 0.5459 | 0.6320 | 0.8571 | 0.2000 | 0.5300 | 0.5762 |
| visual_only | 0.1857 | 0.2124 | 0.4143 | 0.1176 | 0.0000 | 0.2364 | 0.3889 |
| guided_fusion | 0.5104 | 0.4715 | 0.6490 | 0.4848 | 0.5263 | 0.4156 | 0.6147 |
| morphology_fusion | 0.6293 | 0.5955 | 0.6557 | 0.8571 | 0.6250 | 0.4156 | 0.6194 |
| localized_fusion | 0.6495 | 0.6158 | 0.6566 | 0.8571 | 0.7059 | 0.4156 | 0.6194 |
| segment_guided | 0.5956 | 0.5994 | 0.6821 | 0.4615 | 0.6667 | 0.6829 | 0.5712 |
| segment_fusion | 0.6808 | 0.6846 | 0.6838 | 0.7692 | 0.7000 | 0.6829 | 0.5712 |
| segment_visual_bounds | 0.6808 | 0.6846 | 0.6838 | 0.7692 | 0.7000 | 0.6829 | 0.5712 |
| segment_pair_repair | 0.7953 | 0.7971 | 0.8935 | 0.8571 | 0.7619 | 0.6829 | 0.8791 |

父视觉失败：当前主方法输入0次，旧完整提示0次，纯视觉3次；全部保留，无重试。
