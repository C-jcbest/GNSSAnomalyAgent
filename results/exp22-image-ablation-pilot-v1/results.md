# EXP22：SDR暂态修复的同轮有图/去图复核

pilot，2站；已见数据事后诊断。

| 方法 | 暂态Aff-F1 | 暂态事件F1 .5 | 暂态日F1 | 暂态负站FP日 | 四类日F1 |
|---|---:|---:|---:|---:|---:|
| flash_image | 1.0000 | 1.0000 | 1.0000 | 0 | 0.5552 |
| flash_noimage | 1.0000 | 1.0000 | 1.0000 | 0 | 0.5552 |
| vl_image | 1.0000 | 1.0000 | 1.0000 | 0 | 0.5552 |
| vl_noimage | 1.0000 | 1.0000 | 1.0000 | 0 | 0.5552 |
| historical_sdr | 1.0000 | 1.0000 | 1.0000 | 0 | 0.5552 |
| segment_numeric | 0.5000 | 0.6667 | 0.5352 | 0 | 0.4806 |

配对比较：{'flash': {'affiliation_f1_difference': 0.0, 'event_f1_difference': 0.0, 'matched_event_difference': 0}, 'vl': {'affiliation_f1_difference': 0.0, 'event_f1_difference': 0.0, 'matched_event_difference': 0}}；扩量条件：False。
