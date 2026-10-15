# EXP22：SDR暂态修复的同轮有图/去图复核

remaining，48站；已见数据事后诊断。

| 方法 | 暂态Aff-F1 | 暂态事件F1 .5 | 暂态日F1 | 暂态负站FP日 | 四类日F1 |
|---|---:|---:|---:|---:|---:|
| flash_image | 0.9357 | 0.8750 | 0.9346 | 5 | 0.8618 |
| flash_noimage | 0.9357 | 0.8750 | 0.9346 | 5 | 0.8618 |
| vl_image | 0.8125 | 0.8667 | 0.9098 | 5 | 0.8469 |
| vl_noimage | 0.8125 | 0.8667 | 0.9098 | 5 | 0.8469 |
| historical_sdr | 0.8749 | 0.9032 | 0.9372 | 5 | 0.8652 |
| segment_numeric | 0.7500 | 0.8571 | 0.8788 | 0 | 0.8323 |

配对比较：{'flash': {'affiliation_f1_difference': 0.0, 'event_f1_difference': 0.0, 'matched_event_difference': 0}, 'vl': {'affiliation_f1_difference': 0.0, 'event_f1_difference': 0.0, 'matched_event_difference': 0}}；扩量条件：False。
