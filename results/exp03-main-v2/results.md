# GNSS-VLM-EXP03-v2：GNSS排序候选视觉确认、可修正活动边界与训练规模对照实验

范围：test-seed2026101207-offline-recovery。主方法ranked_boundary_fusion，旧主方法anchored_fusion；相同视觉响应派生，非独立调用。

| 批次 | 训练站 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| exp03-main-v2-recovered | 96 | random_forest | 0.5362 | 0.5634 | 0.0000 | 0.4000 | 0.0000 | 0/24 |
| exp03-main-v2-recovered | 96 | rf_balanced | 0.5589 | 0.5273 | 0.0000 | 0.7143 | 0.2857 | 0/24 |
| exp03-main-v2-recovered | 96 | extra_trees | 0.5868 | 0.5475 | 0.0000 | 0.4000 | 0.0000 | 0/24 |
| exp03-main-v2-recovered | 96 | isolation_forest | 0.4215 | — | — | — | — | 0/24 |
| exp03-main-v2-recovered | 96 | rf_temporal | 0.6192 | 0.5718 | 0.4444 | 0.7143 | 0.2857 | 0/24 |
| exp03-main-v2-recovered | 96 | rf_split_temporal | 0.6189 | 0.5718 | 0.4444 | 0.6667 | 0.3077 | 0/24 |
| exp03-main-v2-recovered | 96 | visual_only | 0.4215 | 0.3905 | 0.2727 | 0.1429 | 0.0000 | 1/24 |
| exp03-main-v2-recovered | 96 | fusion | 0.3901 | 0.3449 | 0.2500 | 0.1538 | 0.0000 | 1/24 |
| exp03-main-v2-recovered | 96 | guided_fusion | 0.7899 | 0.7549 | 0.8000 | 0.4667 | 0.5882 | 0/24 |
| exp03-main-v2-recovered | 96 | guided_no_local | 0.5969 | 0.5841 | 0.5556 | 0.5926 | 0.3810 | 0/24 |
| exp03-main-v2-recovered | 96 | guided_no_scores | 0.5440 | 0.5380 | 0.5455 | 0.5556 | 0.2105 | 2/24 |
| exp03-main-v2-recovered | 96 | anchored_fusion | 0.6789 | 0.6160 | 0.5714 | 0.6667 | 0.1818 | 0/24 |
| exp03-main-v2-recovered | 96 | boundary_fusion | 0.7738 | 0.7301 | 0.8571 | 0.6667 | 0.1818 | 0/24 |
| exp03-main-v2-recovered | 96 | boundary_numeric_points | 0.7741 | 0.7301 | 0.8571 | 0.6667 | 0.3077 | 0/24 |
| exp03-main-v2-recovered | 96 | ranked_boundary_fusion | 0.7728 | 0.7301 | 0.8571 | 0.5185 | 0.6250 | 0/24 |
| exp03-main-v2-recovered | 96 | ranked_no_local | 0.6535 | 0.6499 | 0.6154 | 0.6154 | 0.5333 | 0/24 |
| exp03-main-v2-recovered | 96 | ranked_no_scores | 0.7109 | 0.7391 | 0.7143 | 0.7692 | 0.3077 | 2/24 |
| exp03-main-v2-recovered | 96 | visualtime_gnss | 0.0000 | — | — | — | — | 24/24 |

## 成对差值

形变日micro-F1，种子内站点bootstrap 2000次；条件于已拟合模型。

- ranked_boundary_fusion − anchored_fusion：+0.1142，95%CI [-0.0293, 0.2368]。
- ranked_boundary_fusion − boundary_fusion：+0.0000，95%CI [0.0000, 0.0000]。
- ranked_boundary_fusion − guided_fusion：-0.0248，95%CI [-0.1265, 0.0000]。
- ranked_boundary_fusion − rf_split_temporal：+0.1584，95%CI [0.0275, 0.2867]。
- ranked_boundary_fusion − random_forest：+0.1667，95%CI [0.0451, 0.3083]。
- ranked_boundary_fusion − rf_balanced：+0.2029，95%CI [0.0635, 0.3524]。
- ranked_boundary_fusion − fusion：+0.3852，95%CI [0.1573, 0.5940]。
- ranked_boundary_fusion − ranked_no_local：+0.0802，95%CI [-0.0189, 0.2932]。
- ranked_boundary_fusion − ranked_no_scores：-0.0089，95%CI [-0.0474, 0.0440]。
- ranked_boundary_fusion − boundary_numeric_points：+0.0000，95%CI [0.0000, 0.0000]。

全部四类型、±3日辅助及正常误报见CSV；缺测排除，无point adjustment。
开发验证不能充当未见测试；同分布仿真不能证明现场效果。完整论文复现或未运行方法不填成绩。

## 作者源码与适配结果

相同新24测试站，实际使用锁定的VisualTimeAnomaly原prompt/request函数；
qwen模型替换、NEU各point/range六路合并、全部六路成功政策均为GNSS适配。

| 结果标识 | 解析来源 | 通用日F1 | 失败站 |
|---|---|---:|---:|
| visualtime_gnss | 本项目严格解析适配 | 0.0000 | 24/24 |
| visualtime_gnss_author_parser | 锁定作者原解析/向量函数，同响应重放 | 0.2662 | 18/24 |

这是作者源码适配的实际结果，不能视作原论文性能复现。原解析结果详见
[原函数重放](../exp03-author-parser/summary.json)，没有新增请求。
主批因会话中断停止，s0070作者子请求有4份已保存receipt，U-point可能在途、
U-range未完成；该站失败零检测，未重试。其余本项目120份视觉结果全部已保存。
在新恢复目录仅离线派生融合/评分/绘图，原运行未覆盖；
[恢复记录](../../runs/exp03-main-v2-recovered/recovery.json)保留来源，不计第二个独立实验。
