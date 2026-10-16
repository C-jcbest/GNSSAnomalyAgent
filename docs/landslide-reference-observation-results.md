# 训练年参照的证据筛选与附加视觉观察开发试验

日期：2026-10-09；结果：**通过开发筛选，仅可进入完整流程开发**。本批没有新增记录；八个目标来自两个已见背景，不能称独立确认、八个独立背景或完整Agent效果。未升级历史默认流程。

## 机制与两次独立登记

先仅从train中的case2001／2007／2037／2043拟合固定365.25日单谐波，三分量共六系数；基函数仅按当前日期／缺测投影掉常数和线性项以匹配半合成预处理，应用不拟合当前位移、不读取当前注入前背景。引用的是离线分组训练参照，部分train日期晚于验证／测试，不能称严格过去历史预测或在线校正。无注入训练参照是基准已知条件，不是独立专家认定的现场静稳标签；季节变化也可能是真实运动。

原0.03mm/day、2MAD、21／42尺度和原始逐日噪声代理不改。以残差替换触发证据的登记条件为两验证quiet都减少阳性支撑，且两强后弱记录均不损失弱支撑。case2013／2019触发338→51、436→71，但case2024弱支撑168→136，丢32日；普通弱记录也155→96、160→97。故该分支`allows_visual_probe=false`，保持停止，不扫周期或门槛救分。数学上保留注入不保证阈值支撑或视觉可辨识性保持。

| 验证记录 | 家族 | 原阳性支撑 | 残差阳性支撑 | 原弱支撑 | 残差弱支撑 | 弱分母 |
|---|---|---|---|---|---|---|
| case_2013 | quiet | 338 | 51 | 0 | 0 | 0 |
| case_2014 | instrument_step | 353 | 67 | 0 | 0 | 0 |
| case_2015 | weak_constant | 341 | 104 | 155 | 96 | 220 |
| case_2016 | asd | 424 | 246 | 0 | 0 | 0 |
| case_2017 | reactivation | 475 | 317 | 0 | 0 | 0 |
| case_2018 | weak_after_strong | 435 | 371 | 88 | 124 | 210 |
| case_2019 | quiet | 436 | 71 | 0 | 0 | 0 |
| case_2020 | instrument_step | 418 | 95 | 0 | 0 | 0 |
| case_2021 | weak_constant | 443 | 147 | 160 | 97 | 220 |
| case_2022 | asd | 474 | 232 | 0 | 0 | 0 |
| case_2023 | reactivation | 475 | 345 | 0 | 0 | 0 |
| case_2024 | weak_after_strong | 511 | 328 | 168 | 136 | 210 |

另行注册软视觉观察机制，保留原全局／局部图和全部判断权限，参照／残差只作附加观察，不筛选目标、不硬拒绝活动。八目标quiet2025／2031、step2026／2032、普通弱2027／2033、强后弱2030／2036；前六目标180≤day<360，强后弱450≤day<630。三组raw重复、reference_curve叠参照、reference_residual叠参照＋残差，共两遍48请求。每组相同三图、提示、全局／局部原图、补图尺寸和轴限；无重试。目标由开发者指定，不能称自动选窗。参照同一训练模型，阶段头同本半合成训练／验证模型，A/S/D只在活动内。

请求前冻结通过条件：**两遍主残差观察均严格提高目标F1、弱召回不降、任何目标不损失TP、每个quiet／step目标FP不增加，且无契约失败**。未通过不调提示／门槛，不能事后以平均或单遍代替。结果：

```json
{
  "repetitions": [
    {
      "repeat": 1,
      "f1_difference": 0.022032670797457132,
      "weak_recall_difference": 0.030030030030029964,
      "maximum_target_lost_tp": 0,
      "maximum_negative_fp_increase": 0,
      "passes": true
    },
    {
      "repeat": 2,
      "f1_difference": 0.056562743520832626,
      "weak_recall_difference": 0.05255255255255259,
      "maximum_target_lost_tp": 0,
      "maximum_negative_fp_increase": 0,
      "passes": true
    }
  ],
  "proceed_to_complete_window_development": true,
  "scope": "Viewed selected-target development; distinct from the stopped residual-trigger branch"
}
```

## 同目标活动、事件、边界与阶段

九原强数值／学习基线用本半合成共同train24／validation12模型重新应用完整输入；72份预测与冻结缓存一致。再按相同八目标标签评分，共15配置。数值控制未得到新增年参照，**仅作原信息基线背景，不能凭此声称信息公平的完整方法优势**；若后续开发完整方法，须给规则和增广XGBoost同参照并在共同train／validation重新校准。目标截断事件／边界不能代替完整活动边界评估；P无预测时保持未定义，未知计入漏检，不筛掉。

| 配置 | P | R | F1 | 弱R | TP | FP | 事件F1/.3 | 事件F1/.5 | 覆盖 |
|---|---|---|---|---|---|---|---|---|---|
| reference_residual_r0 | 0.4410 | 0.5270 | 0.4802 | 0.5270 | 351 | 445 | 0.4000 | 0.4000 | 0.8742 |
| reference_curve_r0 | 0.3805 | 0.4970 | 0.4310 | 0.4970 | 331 | 539 | 0.3333 | 0.3333 | 1.0000 |
| raw_r0 | 0.4249 | 0.4970 | 0.4581 | 0.4970 | 331 | 448 | 0.4000 | 0.4000 | 0.8742 |
| reference_residual_r1 | 0.4507 | 0.5495 | 0.4953 | 0.5495 | 366 | 446 | 0.6000 | 0.4000 | 0.8742 |
| reference_curve_r1 | 0.5176 | 0.7297 | 0.6056 | 0.7297 | 486 | 453 | 0.6000 | 0.6000 | 0.8742 |
| raw_r1 | 0.3926 | 0.4970 | 0.4387 | 0.4970 | 331 | 512 | 0.3636 | 0.3636 | 0.8742 |
| robust_rule | — | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 |
| pelt_rule | 0.7198 | 0.4745 | 0.5719 | 0.4745 | 316 | 123 | 0.1111 | 0.1111 | 0.9951 |
| xgboost | 0.5764 | 0.1246 | 0.2049 | 0.1246 | 83 | 61 | 0.2222 | 0.0000 | 1.0000 |
| robust_xgboost_stage | — | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 |
| raw_multiscale_rule | 0.7108 | 0.1772 | 0.2837 | 0.1772 | 118 | 48 | 0.2222 | 0.0000 | 1.0000 |
| raw_rule_84 | 0.6188 | 0.1877 | 0.2880 | 0.1877 | 125 | 77 | 0.2500 | 0.2500 | 1.0000 |
| xgboost_raw_persistence | 0.7597 | 0.1471 | 0.2465 | 0.1471 | 98 | 31 | 0.2857 | 0.2857 | 1.0000 |
| xgboost_42_matched_support | 0.7222 | 0.1171 | 0.2016 | 0.1171 | 78 | 30 | 0.0000 | 0.0000 | 1.0000 |
| xgboost_84 | 0.5407 | 0.2192 | 0.3120 | 0.2192 | 146 | 124 | 0.1667 | 0.1667 | 1.0000 |

主残差组F1 .4802／.4953，相对raw .4581／.4387，仍低于PELT目标F1 .5719；原信息强数值控制不是新参照公平比较。两遍TP增20／35均来自case2033普通弱运动；FP分别少3／66，前者来自case2032阶跃，后者来自case2031 quiet。强后弱目标没有新增TP，一条整目标未知保留。曲线单独组第一遍多91FP、第二遍在阶跃多2FP，没有通过相同阴性约束，不能按第二遍较高F1择优。

| 配置 | 起点MAE/d | 终点MAE/d | 活动内ASD宏F1 | 端到端固定ASD宏F1 |
|---|---|---|---|---|
| reference_residual_r0 | 23.0000 | 0.0000 | 0.2127 | 0.1594 |
| reference_curve_r0 | 23.0000 | 0.0000 | 0.2037 | 0.1422 |
| raw_r0 | 23.0000 | 0.0000 | 0.2037 | 0.1515 |
| reference_residual_r1 | 55.3333 | 0.0000 | 0.2195 | 0.1650 |
| reference_curve_r1 | 17.0000 | 0.0000 | 0.2677 | 0.2058 |
| raw_r1 | 23.0000 | 0.0000 | 0.2037 | 0.1444 |
| robust_rule | — | — | 0.0000 | 0.0000 |
| pelt_rule | 39.0000 | 0.0000 | 0.0739 | 0.0726 |
| xgboost | 64.0000 | 23.0000 | 0.0474 | 0.0456 |
| robust_xgboost_stage | — | — | 0.0000 | 0.0000 |
| raw_multiscale_rule | 40.0000 | 52.0000 | 0.0832 | 0.0810 |
| raw_rule_84 | 40.0000 | 8.0000 | 0.0832 | 0.0806 |
| xgboost_raw_persistence | 61.0000 | 11.0000 | 0.0593 | 0.0593 |
| xgboost_42_matched_support | — | — | 0.0483 | 0.0483 |
| xgboost_84 | 55.0000 | 1.0000 | 0.0960 | 0.0899 |

全部参考阶段分母与定义类／固定三类macro保存于`stage-metrics.json`；目标没有A／D参考时不能声称改善这些类。共同学习阶段改善也不能称原生VLM阶段推理改进。

## 全部逐目标改写与失败记录

| 遍 | 附加观察 | 记录 | 家族 | 丢TP | 增TP | 撤FP | 增FP | 已知日未知 |
|---|---|---|---|---|---|---|---|---|
| 1 | reference_curve | case_2025 | quiet | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_curve | case_2026 | instrument_step | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_curve | case_2027 | weak_constant | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_curve | case_2030 | weak_after_strong | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_curve | case_2031 | quiet | 0 | 0 | 0 | 91 | 0 |
| 1 | reference_curve | case_2032 | instrument_step | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_curve | case_2033 | weak_constant | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_curve | case_2036 | weak_after_strong | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_residual | case_2025 | quiet | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_residual | case_2026 | instrument_step | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_residual | case_2027 | weak_constant | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_residual | case_2030 | weak_after_strong | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_residual | case_2031 | quiet | 0 | 0 | 0 | 0 | 0 |
| 1 | reference_residual | case_2032 | instrument_step | 0 | 0 | 3 | 0 | 0 |
| 1 | reference_residual | case_2033 | weak_constant | 0 | 20 | 0 | 0 | 0 |
| 1 | reference_residual | case_2036 | weak_after_strong | 0 | 0 | 0 | 0 | 180 |
| 2 | reference_curve | case_2025 | quiet | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_curve | case_2026 | instrument_step | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_curve | case_2027 | weak_constant | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_curve | case_2030 | weak_after_strong | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_curve | case_2031 | quiet | 0 | 0 | 66 | 0 | 0 |
| 2 | reference_curve | case_2032 | instrument_step | 0 | 0 | 0 | 2 | 0 |
| 2 | reference_curve | case_2033 | weak_constant | 0 | 155 | 0 | 5 | 0 |
| 2 | reference_curve | case_2036 | weak_after_strong | 0 | 0 | 0 | 0 | 180 |
| 2 | reference_residual | case_2025 | quiet | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_residual | case_2026 | instrument_step | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_residual | case_2027 | weak_constant | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_residual | case_2030 | weak_after_strong | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_residual | case_2031 | quiet | 0 | 0 | 66 | 0 | 0 |
| 2 | reference_residual | case_2032 | instrument_step | 0 | 0 | 0 | 0 | 0 |
| 2 | reference_residual | case_2033 | weak_constant | 0 | 35 | 0 | 0 | 0 |
| 2 | reference_residual | case_2036 | weak_after_strong | 0 | 0 | 0 | 0 | 180 |

v1在准备时raw补图与全局原图重名，覆盖全局图。发现后立即中断，30已启动请求、29完成回执、128945已知token全部保留；一条中断请求无响应，token未知。该批不评分、不用于效果结论。v2为独立源码／目录修复图名，准备时检查三文件不同及原全局图与源图逐字节相同，协议未按无效回答调参。旧v1源码、预测、回执和failure-freeze不改。年度证据v1标签人口不匹配的0API失败也保留，v2使用显式标签子集。

## 成本、核验与复现

有效配对批48请求／246220token；0契约失败。累计1485启动、1484完成回执／**至少6533182token**，包含无效批成本；一条中断请求费用未知，不能报精确总账。原96受控与60半合成分开计数；这次复用八记录，不新增样本。补图不产生额外API，但增加图像token和绘图成本。

| 观察/遍 | 请求 | token | 回执秒数和 |
|---|---|---|---|
| reference_residual_r0 | 8 | 41029 | 29.3750 |
| reference_curve_r0 | 8 | 41064 | 24.1710 |
| raw_r0 | 8 | 41024 | 25.1710 |
| reference_residual_r1 | 8 | 41070 | 31.1390 |
| reference_curve_r1 | 8 | 41020 | 26.3910 |
| raw_r1 | 8 | 41013 | 25.2660 |

冻结模型／输入／源码及48实际payload绑定，15活动／阶段指标独立核验、72数值重算、32改写守恒、8目标三图／同提示和8图导出通过；四训练参照系数精确重构。2年度参照业务测试与Ruff通过。模型加载／数值预测单次本机1.500秒，排除特征、工具绘图、拟合和服务，不报告能源。硬筛选失败与软观察通过均为本开发条件下的结论，不推广为一般年周期方法的有效性判断。第一次补充核验因已有checker未保留无预测时的Precision=None失败，0API；独立v2改用既有支持未定义Precision的核验函数，并将全部控制掩码到相同目标，失败源保留。

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
$taskPython='artifacts/reproduction-2026-10-07/environment/Scripts/python.exe'
# 以下输出目录必须是新的，不重跑已完成推理
& $taskPython scripts/run_landslide_reference_observation_probe_v2.py prepare --out artifacts/reference-probe-reproduction
& $taskPython scripts/run_landslide_reference_observation_probe_v2.py run --out artifacts/reference-probe-reproduction
& $taskPython scripts/run_landslide_reference_observation_probe_v2.py analyze --out artifacts/reference-probe-reproduction --analysis artifacts/reference-probe-reproduction/analysis
& $taskPython scripts/summarize_landslide_reference_observation_v2.py --run artifacts/reference-probe-reproduction --analysis artifacts/reference-probe-reproduction/analysis --out artifacts/reference-probe-reproduction/analysis/supplement
```

产物：`annual-reference-evidence-v2`、`reference-observation-development-v2/analysis-v1`及`supplement-v1`；失败v1完整保留。最近邻继续沿用已核查MORA双视图／受约束纠正及TimeRLM视觉工具源码；不将年参照、双图或工具包装本身当新贡献。用户暂无领域复核者，继续自主研究，不重复索要人员或造专家标签。
