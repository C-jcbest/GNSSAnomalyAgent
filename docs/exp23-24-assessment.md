# EXP23—24：保留Affiliation主评价，继续优化候选恢复

2026-10-15。分支`codex/exp23-affiliation-candidate-recovery`，父版本6076f59，
开始前备份标签`backup/pre-exp23-affiliation-retained-20261015-6076f59`。

**当前决策：保留SDR-Flash-Aff，并保留SDR-GR缺测边界恢复为补充候选。**
Affiliation P/R/F1是后续范围任务主评价，IoU仅作补充。新方案在旧站确有增益；
新48验证站再次支持原Flash语义修复路径，尚未证明新增恢复机制普遍有效。
不要求所有指标同时提高，不因IoU未升取消Affiliation正结果。

## 文献评价口径

1. Xu等，*Can Multimodal LLMs Perform Time Series Anomaly Detection?*，
   [arXiv:2502.17812v1](https://arxiv.org/html/2502.17812v1)，§3.4 Evaluation Metrics：
   对point-wise与range-wise明确使用Huet等Affiliation P/R/F1。正式记录为
   [ACM Web Conference 2026，DOI](https://doi.org/10.1145/3774904.3792376)，
   作者README的WWW25年份冲突仍保留，不能照抄。
2. Zhou、Yu，*Can LLMs Understand Time Series Anomalies?*（AnomLLM），ICLR2025，
   [arXiv:2410.05440v3](https://arxiv.org/html/2410.05440v3)，§5.1 Metrics：
   LLM输出离散区间，以affinity P/R的调和F1为主要指标，不直接使用需要连续评分的VUS。
   本轮成功读取[作者notebook及utils](https://github.com/rose-stl-lab/anomllm)，
   固定tree `bf7827dfd63e5cd134f5b2412efc54175e5aee72`；
   notebook逐序列Aff-F1后均值，`compute_metrics`在双空时给1、单空时给0，先保留三位小数。
3. Huet、Navarro、Rossi，*Local Evaluation of Time Series Anomaly Detection Algorithms*，
   [KDD2022 DOI](https://doi.org/10.1145/3534678.3539339)。继续执行已锁定作者原Affiliation源码，
   使用EXP20明确披露的正样本站均值、缺测/空集适配；不把跨站约定说成所有论文统一标准。
4. TAMA，*See it, Think it, Sorted*，[arXiv:2411.02465v1](https://arxiv.org/html/2411.02465v1)，
   Multi-scaled Self-reflection：提供新局部图复核候选。本轮借其局部复核思想，
   不复制置信阈值与point adjustment，不声称完整作者原法复现。

以上支持采用Affiliation主评价，但不能推断所有范围异常论文都不用IoU。
来源全文继续复用已缓存证据，新增作者代码、API定位和SHA保存在
`papers/exp23-affiliation-candidate-recovery/`，原文件未修改、MIT许可保留。
AnomLLM双空给1与本项目无GT站Aff=null、另列误报不同，绝对分数不能直接横比。

## 方法名称与结果键

| 名称 | 实验/结果键 | 作用及状态 |
|---|---|---|
| SDR-Flash-Aff | EXP22 `flash_image`；本轮`base_flash` | 原SDR＋qwen3.8-flash，正式保留为当前研究候选 |
| SDR-CR | EXP23 `flash_image` / `vl_image` | 恢复被阶跃筛选舍弃的PELT起点，再局部图复核；原始版本保留为探索分支 |
| SDR-GR | EXP24 `gap_flash_image` / `gap_vl_image` | 在SDR-CR响应后限定缺测后首个观测边界；保留为特殊场景补充候选 |
| 数值恢复 | EXP23 `recovery_numeric` / EXP24 `gap_numeric` | 同候选、原returning_pair信息准则判别；标为本项目适配 |
| 全接纳候选 | `candidate_all` / `gap_candidate_all` | 同候选简单诊断基线，不把其成绩算作模型贡献 |

原数值`segment_numeric`、Flash/VL各自有图/去图始终保留。生产源码和25页EXP06 Word
未替换。文稿后续应区分原冻结主实验、事后范围评价和本轮新增开发验证。

## EXP23：从真实漏检调用关系提出候选恢复

代码探索发现：PELT输出的分段边界不一定成为jump。原1406/s0063在起点前一日缺测，
`segment_proposals`的观测支持约束跳过该jump，原SDR仅看到返回step，无法形成完整候选。
新方案恢复被舍弃起点，与仍未配对的step构成范围；沿用±3日细化去重和120日最大间隔，
不改位移/速率阈值。模型只选候选ID，不自由猜日边界。图保留原N/E/U和正常历史残差、缺测。
持续形变范围/阶段不改。详见[调用前登记](experiment-exp23.md)。

先在原48站诊断，只有1站触发，4个实际调用。Flash与VL有图均接受473—519，去图均不接受。
同候选数值判别也接受，证明这例恢复有用，但不证明VLM优于数值恢复。

旧48站，每种方法16个暂态正样本站，其他类型负站误报另列。

| 方法 | Aff-P | Aff-R | 站均Aff-F1 | 日F1 | 覆盖 | 总FP日 |
|---|---:|---:|---:|---:|---:|---:|
| 原数值 | .7500 | .7500 | .7500 | .8788 | .7500 | 0 |
| SDR-Flash-Aff | .9339 | .9375 | .9357 | .9346 | .9375 | 26 |
| SDR-CR Flash/VL有图 | .9964 | 1.0000 | .9982 | .9777 | 1.0000 | 26 |
| SDR-CR Flash/VL去图 | .9339 | .9375 | .9357 | .9346 | .9375 | 26 |
| 同候选数值恢复/全接纳 | .9964 | 1.0000 | .9982 | .9777 | 1.0000 | 26 |

新增47个TP日、无新增FP日；本轮请求图像差值在这个旧站上实际传到最终范围。
这是1个已见漏检启发的开发案例，不能外推为总体视觉优势或新的独立测试。

## 新1516验证48站：原Flash有效，宽泛恢复有误报取舍

按照预先登记生成96训练/48验证/48test。只执行validation，test未评分；
原Flash修复有5触发站5请求，候选恢复有2触发站8请求，pilot已涵盖全部恢复触发站，
因此没有额外恢复扩量调用。注册、输入、失败、原响应均保存，不挑站、不重试。

| 方法 | Aff-P | Aff-R | 站均Aff-F1 | 日F1 | 覆盖 | 总FP日 |
|---|---:|---:|---:|---:|---:|---:|
| 原数值 | .6250 | .6250 | .6250 | .7513 | .6250 | 0 |
| SDR-Flash-Aff | .8750 | .8585 | .8663 | .8982 | .8006 | 0 |
| SDR-CR Flash有图 | .8641 | .8585 | .8603 | .8254 | .8006 | 78 |
| SDR-CR Flash去图 | .8750 | .8585 | .8663 | .8982 | .8006 | 0 |
| SDR-CR VL有图 | .9101 | .9089 | .9084 | .7948 | .8006 | 115 |
| SDR-CR VL去图 | .9210 | .9089 | .9144 | .8621 | .8006 | 37 |
| 同候选数值恢复 | .8750 | .8585 | .8663 | .8982 | .8006 | 0 |
| 全接纳候选 | .9101 | .9089 | .9084 | .7948 | .8006 | 115 |

原Flash相对数值Aff-F1 +.2413，日F1 +.1469，无新增暂态FP日，支持保留原方案。
VL恢复相对Flash Aff-F1 +.0422，满足主要指标正向，但不等于新增真值重叠：
TP日仍397，覆盖仍.8006，增加115个FP日。Affiliation利用历时距离，未重叠但邻近的
预测也可能提高Aff-R；这里不能写成“精确恢复漏检范围”。去图VL更高，且有图等于
全接纳候选，不据此宣称图像优于数值方案。按用户目标保留这个Aff正结果供选择，
不以IoU否决，但将误报取舍和归因事实一并保留。

两个站原始模型理由有与曲线不符、甚至无图却描述所见曲线的内容；这些话不作物理证据。
所有结果以实际pair_ids和固定真值评分，不把模型生成的解释当成人工标注。

## EXP24：缺测边界约束保留旧收益、消除本轮新增误报

新1516的两个恢复起点均在连续观测内，局部曲线呈平滑变化，不能仅因PELT分段就解释成跳变。
针对原问题的代码约束，增加响应后结构限制：只补充start前一日缺测、start本日有观测的
平台起点。这是看过EXP23结果后的机制修正，明确事后开发；复用相同请求的保存响应，0新调用。
不把过滤后的新候选提示假装已经重新请求。详见[EXP24说明](experiment-exp24.md)。

| 数据 | SDR-Flash-Aff Aff-F1 | SDR-GR有图Aff-F1 | 去图Aff-F1 | 日F1（原→GR） | 总FP日（原→GR） |
|---|---:|---:|---:|---:|---:|
| 旧48站开发分析 | .9357 | .9982 | .9357 | .9346→.9777 | 26→26 |
| 新1516验证48站上的事后修正 | .8663 | .8663 | .8663 | .8982→.8982 | 0→0 |

新1516没有满足缺测恢复条件的候选，结果不变是“无干预机会”，不能说独立验证了恢复收益。
旧站仍仅1个正例，数值恢复同分。因此保留SDR-GR为补充候选，而非声称已全面优于其他方法。
原始SDR-CR VL的Aff正结果也不删除，不混入GR较低误报结果冒充同一方法。

两轮长期形变Aff-F1分别.9041/.9452，所有修复方法与数值完全相同。
旧站范围并集Aff-F1从Flash .9480到GR .9508，新1516维持.9714；主要收益仍是暂态类别修复。

## 当前采用与下一项实验

- 当前研究主候选保留SDR-Flash-Aff；新增SDR-GR作为缺测边界补充路径，结果与状态分开登记。
- 后续优先收集或冻结新的“异常起点紧邻缺测”的实例，比较原Flash、GR视觉、GR数值、去图。
  可在简单原模拟框架中单独做缺测位置敏感性实验，但需标注为受控条件实验，不能当自然总体数据。
- 另一方向是对恢复出的区间做“边界发生了突变还是平滑弯曲”的局部存在性核验，
  重点控制新1516暴露的假平台，不再盲目堆DPI或让模型自由读日级日期。
- 本轮尚未执行新的独立缺测专门试验，不预报成绩。目标是证明有限GNSS条件下的实用增量，
  不要求所有数据、所有指标或所有优秀论文方法上全面领先。

## 验证、证据与复算

EXP23实际17请求：旧站4、新Flash基线5、新恢复8；17 HTTP200，0模型调用失败，0重试。
旧站第4个响应已保存后终端GBK打印Unicode失败，核对4份result/response后补完成回执，
没有重发请求。最初prepare v1因字段名错误在任何请求前失败；有效输入为historical-v2。
EXP24为0请求的离线派生。原始失败、旧结果没有覆盖。

新增缺测约束测试连同本轮3项候选恢复测试共4项通过；最终项目64项测试及
本轮脚本静态检查全部通过。范围指标复用既有作者60项测试已验证实现，没有重复改写作者源码。
本轮围绕图/缺测、边界重复、保留活动、响应不可变做业务验证。

复算不调用模型：

```powershell
uv run --no-sync python scripts/experiment_exp23.py summarize --visual runs/exp23-historical-recovery-v2 --out results/NEW-historical --phase full
uv run --no-sync python scripts/experiment_exp23.py summarize --visual runs/exp23-development-1516-recovery-v1 --out results/NEW-validation --phase full
uv run --no-sync python scripts/experiment_exp24.py --source runs/exp23-historical-recovery-v2 --out results/NEW-gap-historical
uv run --no-sync python scripts/experiment_exp24.py --source runs/exp23-development-1516-recovery-v1 --out results/NEW-gap-validation
```

新数据可用已登记seed2026101516、per-split48、train-stations96重生成。
所有范围指标与逐站表分别位于`results/exp23-*`、`results/exp24-*`。
[汇总图](../results/exp23-24-assessment/affiliation-and-errors.png)、
[逐请求及完整指标证据](../results/exp23-24-assessment/evidence.json)及该目录的原响应/图像/冻结源码保留。
完整本地备份与SHA见`docs/exp23-24-backup.json`；不含.env凭据。
