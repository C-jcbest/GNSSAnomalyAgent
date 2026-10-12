# GNSS-VLM-EXP01：基于站点正常背景的GNSS视觉数值融合异常识别仿真对照与消融实验

协议版本：`v4`。名称和编号于2026-10-12补充；原冻结登记、实验条件和结果未修改。

## 实验批次

| 运行目录 | 名称 |
|---|---|
| `main-v4` | 主实验：数值对照、视觉数值融合与信息组消融 |
| `main-repeat-v4` | 同测试集复测：默认RF、纯视觉与融合 |
| `main-upstream-repeat-v4` | 作者源码适配的独立响应复测 |

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run（用户授权）
- Origin Date: 2026-10-12
- Verification Status: 保存响应的确定性离线评分；不是现场验证或完整论文复现。
- Version Label: gnss-vlm-research-v4

## 范围

模型：`qwen3.8-flash`。24独立测试站；训练与验证站分别独立。无统一物理阈值、无测试阈值扫描、无point adjustment。
所有表格由脚本读取保存结果生成。作者源码只提供通用异常，形变列留空；rf_balanced是特别标注的class_weight调整，默认RF单列。

| 方法 | 通用异常日F1 | 持续形变日F1 | 形变事件F1 IoU≥0.5 | 正常站误报段/站年 | 失败/未执行 |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.4436 | 0.5321 | 0.0645 | 0.2558 | 0/0 |
| rf_balanced | 0.6040 | 0.5789 | 0.0193 | 0.5116 | 0/0 |
| extra_trees | 0.5037 | 0.5046 | 0.0606 | 0.0000 | 0/0 |
| isolation_forest | 0.4979 | — | — | 17.1371 | 0/0 |
| visual_only | 0.4225 | 0.3673 | 0.3529 | 6.1387 | 7/0 |
| fusion | 0.3940 | 0.3900 | 0.3636 | 3.3251 | 5/0 |
| fusion_no_reference | 0.3910 | 0.3724 | 0.5000 | 2.8136 | 6/0 |
| fusion_n_only | 0.4077 | 0.3687 | 0.4800 | 2.8136 | 5/0 |
| visualtime_gnss | 0.0000 | — | — | 0.0000 | 24/0 |
| visualtime_gnss_author_parser | 0.2809 | — | — | 1.2789 | 19/0 |
| visualtime_gnss_author_parser_repeat | 0.1013 | — | — | 0.0000 | 19/0 |

`visualtime_gnss`为原提示＋GNSS输入＋本项目严格解析；`visualtime_gnss_author_parser`为同批保存响应调用作者原解析和向量转换。后者是主实验登记后的单列离线分析，无新增请求、提示修改或阈值调优。原代码的列表截取、截断补括号、索引跳过/裁剪行为均保留；两列都要求六个子请求全部成功，否则整站记失败和零检测。具体源文件哈希与每个子响应解析结果保存在author-parser-analysis中。

作者源码复测列来自同24站的另一批144次真实调用，其请求与首批逐一核对。主批有43次连接错误及3次读取错误，因此复测用于区分接口可靠性与检测表现。原批、失败与复测同时保留，没有以高分批次替换首批，也没有增加独立站点数。新批训练过程和源码快照独立保存；提示、默认参数和输入未改变。

## 分类型表现

逐日P/R/F1包含失败站的漏检；事件F1为一对一匹配、IoU≥0.5。尖峰/阶跃真值是发生日期，永久阶跃后的平台不计为每日异常。

| 方法 | 类型 | 日P | 日R | 日F1 | 事件F1 |
|---|---|---:|---:|---:|---:|
| random_forest | spike | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_forest | step | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_forest | transient | 0.0508 | 0.0833 | 0.0631 | 0.0000 |
| random_forest | deformation | 0.6955 | 0.4309 | 0.5321 | 0.0645 |
| rf_balanced | spike | 0.6000 | 0.4286 | 0.5000 | 0.5000 |
| rf_balanced | step | 0.5000 | 0.2500 | 0.3333 | 0.3333 |
| rf_balanced | transient | 0.1203 | 0.2065 | 0.1520 | 0.0000 |
| rf_balanced | deformation | 0.4958 | 0.6956 | 0.5789 | 0.0193 |
| extra_trees | spike | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| extra_trees | step | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| extra_trees | transient | 0.0656 | 0.0145 | 0.0237 | 0.0000 |
| extra_trees | deformation | 0.6603 | 0.4084 | 0.5046 | 0.0606 |
| visual_only | spike | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| visual_only | step | 0.1429 | 0.1250 | 0.1333 | 0.1176 |
| visual_only | transient | 0.7945 | 0.2101 | 0.3324 | 0.2222 |
| visual_only | deformation | 0.2642 | 0.6024 | 0.3673 | 0.3529 |
| fusion | spike | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fusion | step | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fusion | transient | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fusion | deformation | 0.2654 | 0.7353 | 0.3900 | 0.3636 |
| fusion_no_reference | spike | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fusion_no_reference | step | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fusion_no_reference | transient | 0.8182 | 0.1630 | 0.2719 | 0.2222 |
| fusion_no_reference | deformation | 0.2636 | 0.6345 | 0.3724 | 0.5000 |
| fusion_n_only | spike | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fusion_n_only | step | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fusion_n_only | transient | 0.8261 | 0.2754 | 0.4130 | 0.4000 |
| fusion_n_only | deformation | 0.2567 | 0.6538 | 0.3687 | 0.4800 |

## 配对不确定性

以下为整站macro-F1差值的2000次成对bootstrap，和上表micro-F1不是同一统计量。区间只描述该仿真样本，不证明现场有效性或首次创新。

| 比较 | 差值 | 95%百分位区间 |
|---|---:|---:|
| fusion_minus_random_forest | -0.0279 | [-0.1669, 0.1073] |
| fusion_minus_random_forest_deformation | 0.0411 | [-0.0722, 0.1506] |
| fusion_minus_rf_balanced | -0.1289 | [-0.2447, -0.0214] |
| fusion_minus_rf_balanced_deformation | -0.0232 | [-0.1124, 0.0474] |
| fusion_minus_extra_trees | -0.0032 | [-0.1263, 0.1166] |
| fusion_minus_extra_trees_deformation | 0.0477 | [-0.0652, 0.1610] |
| fusion_minus_isolation_forest | -0.1031 | [-0.2105, -0.0039] |
| fusion_minus_visual_only | -0.0547 | [-0.1850, 0.0559] |
| fusion_minus_visual_only_deformation | 0.0495 | [-0.0122, 0.1380] |
| fusion_minus_fusion_no_reference | -0.0461 | [-0.1349, 0.0089] |
| fusion_minus_fusion_no_reference_deformation | -0.0110 | [-0.0390, 0.0145] |
| fusion_minus_fusion_n_only | -0.1188 | [-0.2517, -0.0106] |
| fusion_minus_fusion_n_only_deformation | -0.0178 | [-0.0557, 0.0138] |
| fusion_minus_visualtime_gnss | 0.2061 | [0.0833, 0.3363] |
| fusion_minus_visualtime_gnss_author_parser | 0.1249 | [-0.0165, 0.2765] |

## 同测试集独立响应复测

仅重复推理，不增加独立样本。RF确定性重放，视觉再次真实调用；不选择高分一遍。

| 方法 | 首遍通用F1 | 第二遍通用F1 | 首遍形变F1 | 第二遍形变F1 |
|---|---:|---:|---:|---:|
| random_forest | 0.4436 | 0.4436 | 0.5321 | 0.5321 |
| visual_only | 0.4225 | 0.4075 | 0.3673 | 0.4129 |
| fusion | 0.3940 | 0.3563 | 0.3900 | 0.3326 |

第二遍失败站数：random_forest=0/24；visual_only=4/24；fusion=0/24。

## 代价与限制

```json
{
  "main": {
    "requests": 240,
    "prompt_tokens": 886774,
    "completion_tokens": 48811,
    "total_tokens": 935585,
    "summed_inference_seconds": 1179.9294969015755,
    "failed_request_error_types": {
      "ReadError": 3,
      "ConnectError": 43,
      "ValidationError": 2,
      "JSONDecodeError": 116,
      "ValueError": 2
    },
    "timing_note": "sum of request/record times, not concurrent wall time or billing"
  },
  "run": "C:\\Users\\JC\\.codex\\worktrees\\gnss-vlm-research\\GNSSAnomalyAgent-sim\\runs\\main-v4",
  "experiment": {
    "experiment_id": "GNSS-VLM-EXP01",
    "name": "\u57fa\u4e8e\u7ad9\u70b9\u6b63\u5e38\u80cc\u666f\u7684GNSS\u89c6\u89c9\u6570\u503c\u878d\u5408\u5f02\u5e38\u8bc6\u522b\u4eff\u771f\u5bf9\u7167\u4e0e\u6d88\u878d\u5b9e\u9a8c",
    "protocol_version": "v4",
    "label_added_on": "2026-10-12",
    "label_note": "\u5b9e\u9a8c\u5b8c\u6210\u540e\u8865\u5145\u7684\u7ba1\u7406\u540d\u79f0\u4e0e\u7f16\u53f7\uff0c\u4e0d\u4fee\u6539\u539f\u59cb\u51bb\u7ed3\u767b\u8bb0\u3001\u6761\u4ef6\u6216\u7ed3\u679c\uff0c\u4e5f\u4e0d\u662f\u5df2\u53d1\u8868\u7684\u65b9\u6cd5\u540d\u79f0\u3002",
    "runs": {
      "main-v4": "\u4e3b\u5b9e\u9a8c\uff1a\u6570\u503c\u5bf9\u7167\u3001\u89c6\u89c9\u6570\u503c\u878d\u5408\u4e0e\u4fe1\u606f\u7ec4\u6d88\u878d",
      "main-repeat-v4": "\u540c\u6d4b\u8bd5\u96c6\u590d\u6d4b\uff1a\u9ed8\u8ba4RF\u3001\u7eaf\u89c6\u89c9\u4e0e\u878d\u5408",
      "main-upstream-repeat-v4": "\u4f5c\u8005\u6e90\u7801\u9002\u914d\u7684\u72ec\u7acb\u54cd\u5e94\u590d\u6d4b"
    }
  },
  "registration_sha256": "6c6fe3825f1fa307ff305e6ecc56f1d1fe3a2d75793c2bc6bafe5bbaa762e8da",
  "summary_sha256": "bd136e66f034ee465766535899d628693b0d549c7fab79d7b078d16bfb55c7e4",
  "evaluation_directory": "evaluation-corrected",
  "report_script_sha256": "8ba8d50b12fa991727495cc9ccda8bc62e524f21df3530697c13b41fe598c24c",
  "author_repeat": {
    "requests": 144,
    "prompt_tokens": 74734,
    "completion_tokens": 62121,
    "total_tokens": 136855,
    "summed_inference_seconds": 1146.4250748988707,
    "failed_request_error_types": {
      "JSONDecodeError": 141,
      "ReadTimeout": 1
    },
    "timing_note": "sum of request/record times, not concurrent wall time or billing"
  },
  "author_repeat_analysis_sha256": "4e91d6d38268d96847446ba34bde6b3299dca607d8e47343972f05c2967e19e9",
  "all_recorded_runs": {
    "main": {
      "requests": 0,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0,
      "summed_inference_seconds": 0.0,
      "failed_request_error_types": {},
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "main-repeat": {
      "requests": 0,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0,
      "summed_inference_seconds": 0.0,
      "failed_request_error_types": {},
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "main-repeat-v4": {
      "requests": 48,
      "prompt_tokens": 677056,
      "completion_tokens": 5289,
      "total_tokens": 682345,
      "summed_inference_seconds": 296.38590749981813,
      "failed_request_error_types": {
        "ValueError": 3,
        "ValidationError": 1
      },
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "main-upstream-repeat-v4": {
      "requests": 144,
      "prompt_tokens": 74734,
      "completion_tokens": 62121,
      "total_tokens": 136855,
      "summed_inference_seconds": 1146.4250748988707,
      "failed_request_error_types": {
        "JSONDecodeError": 141,
        "ReadTimeout": 1
      },
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "main-v4": {
      "requests": 240,
      "prompt_tokens": 886774,
      "completion_tokens": 48811,
      "total_tokens": 935585,
      "summed_inference_seconds": 1179.9294969015755,
      "failed_request_error_types": {
        "ReadError": 3,
        "ConnectError": 43,
        "ValidationError": 2,
        "JSONDecodeError": 116,
        "ValueError": 2
      },
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "numerical-smoke": {
      "requests": 0,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0,
      "summed_inference_seconds": 2.5290496998932213,
      "failed_request_error_types": {},
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "paper-pilot": {
      "requests": 0,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0,
      "summed_inference_seconds": 0.0,
      "failed_request_error_types": {},
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "paper-pilot-v4": {
      "requests": 0,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0,
      "summed_inference_seconds": 0.0,
      "failed_request_error_types": {},
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "paper-pilot-v4b": {
      "requests": 36,
      "prompt_tokens": 18810,
      "completion_tokens": 12752,
      "total_tokens": 31562,
      "summed_inference_seconds": 257.5744653000729,
      "failed_request_error_types": {
        "JSONDecodeError": 36
      },
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "pilot": {
      "requests": 24,
      "prompt_tokens": 199124,
      "completion_tokens": 6893,
      "total_tokens": 206017,
      "summed_inference_seconds": 177.5604223004775,
      "failed_request_error_types": {
        "ValidationError": 12
      },
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "pilot-v2": {
      "requests": 24,
      "prompt_tokens": 204510,
      "completion_tokens": 4356,
      "total_tokens": 208866,
      "summed_inference_seconds": 139.94279480038676,
      "failed_request_error_types": {
        "ValidationError": 6
      },
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "pilot-v3": {
      "requests": 24,
      "prompt_tokens": 265839,
      "completion_tokens": 4073,
      "total_tokens": 269912,
      "summed_inference_seconds": 141.82205629965756,
      "failed_request_error_types": {
        "ValueError": 1,
        "ValidationError": 5
      },
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "pilot-v4": {
      "requests": 24,
      "prompt_tokens": 260151,
      "completion_tokens": 2598,
      "total_tokens": 262749,
      "summed_inference_seconds": 130.0372962997062,
      "failed_request_error_types": {},
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    },
    "schema-capability-probe": {
      "requests": 1,
      "prompt_tokens": 20259,
      "completion_tokens": 165,
      "total_tokens": 20424,
      "summed_inference_seconds": 7.5562975000357255,
      "failed_request_error_types": {},
      "timing_note": "sum of request/record times, not concurrent wall time or billing"
    }
  },
  "total_recorded_requests": 565,
  "repeat": {
    "requests": 48,
    "prompt_tokens": 677056,
    "completion_tokens": 5289,
    "total_tokens": 682345,
    "summed_inference_seconds": 296.38590749981813,
    "failed_request_error_types": {
      "ValueError": 3,
      "ValidationError": 1
    },
    "timing_note": "sum of request/record times, not concurrent wall time or billing"
  }
}
```

背景正常前缀是仿真实验提供的条件，真实部署需要独立确认；这是重要适用性限制。只有4个纯正常测试站与8个有形变测试站，事件统计较粗，不能以大量日点数冒充独立样本。质量尖峰/阶跃标签很稀少，默认RF与平衡RF的局限须与视觉一起报告。
缺测排除评分并保留图和导出掩码；缺测会切开数值预测的连续段，事件F1可能因此降低，本轮没有为了提高事件分数而填补或桥接。

本轮扁平视觉契约不输出内部阶段；早期嵌套阶段接口失败开发批次保留。本报告不是可直接提交的论文，方法优势和新颖性不能由单批合成结果保证。
