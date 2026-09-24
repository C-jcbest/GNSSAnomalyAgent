# GNSS 异常实验的文献与代码调研

调研日期：2026-09-22。本记录区分“原文/代码事实”和“本项目设计判断”。重点核对用户指定文献，不声称已穷尽截至该日的全部研究。GNSS 日解、稳健窗口与周期处理专题已补充至 [日周期调研与方案](gnss-daily-preprocessing-review.md)，区分原文事实、摘要证据和本项目提案。

## 1 核心判断

VisualTimeAnomaly 的论文 v2 已在能力研究之后提出 TSAD-Agents，覆盖异常类型扫描、工具与模态规划、检测及复核，是本项目最直接的方法参考。TAMA 提供视觉检测与多尺度复核思路，TSB-AD 提供调参分离和评价规范。用户已明确本研究定位为 GNSS 领域应用，允许采用已有数值视觉复核机制；首版重点验证领域中的方法能力、缺测表现和组合效果，不要求提出新的通用机制。论文明确区分所借鉴的方法与本项目的领域配置及验证结果即可。

TSB-AutoAD 作为自动方法选择的补充阅读。其随机选择、集成、非 LLM 选择器适合后续扩展；按当前简化范围，不全部加入首版实验。

## 2 VisualTimeAnomaly 与 TSAD-Agents [R1]

### 2.1 版本与调研更正

按用户提示重新核对 [arXiv v2 摘要页](https://arxiv.org/abs/2502.17812v2) 及 [v2 全文](https://arxiv.org/html/2502.17812v2)：版本日期为 **2026-02-17**，Comments 标注 **ACM Web Conference 2026（WWW’26）**。作者为 Xiongxiao Xu、Haoran Wang、Yueqing Liang、Philip S. Yu、Yue Zhao、Kai Shu。

主要方法依据为 v2 第 2–6 节、图 8、表 4、图 9 和附录。官方仓库 README 仍写 WWW 2025，与 arXiv v2 不一致，会议与论文方法引用以明确的 v2 记录为准。

### 2.2 能力实验与适用范围

v2 覆盖点、区间和变量级异常，使用合成正弦/余弦及 UCR Symbols、UEA ArticularyWordRecognition 的真实序列背景注入异常。单变量长度为 400，多变量序列长度为 200；每类生成 100 幅不同噪声的图像，实验重复 3 次。其真实序列背景不是实际 GNSS 或滑坡事件。v1 的“12.4k 图像”是旧版明示规模，不直接作为 v2 实验规模引用。

单变量点异常包括 global/contextual，区间异常包括 seasonal/trend/shapelet；多变量主要定位整条异常变量，注入 triangle/square/sawtooth/random 等不同于其他变量的模式，不等于逐时刻的 N/E/U 联合检测。

v2 的 RQ2 在 0%–25%、每档 5% 的随机缺测下，重点比较 global、trend、random 三种类型。构造不规则异常时排除 contextual，因删点破坏其局部上下文。其传统基线采用均值填补以适配缺测，因此相关优劣结论同时受预处理影响。本项目当前采用保留日网格、不插值的缺测感知计算，见[日预实验协议](daily-pilot-protocol.md)，不另设完整预处理消融。

“缺测下分数稳定”也不代表绝对检测质量高：v2 第 4 节举出的点异常视觉平均 F1 在完整与 25% 缺测下分别约为 2.58 和 2.56。这支持同时报告性能水平与缺测退化幅度，不能仅靠相对下降很小论证鲁棒。

v1 明确用 affiliation 评价点/区间异常；v2 第 2.2 节写 precision、recall、F1，不能据此直接继承 v1 的具体计算契约。复建前应分别审计版本、标签处理、阈值与指标实现，不混用两版数值。v2 还增加 iForest、OCSVM、OmniAnomaly、THOC、TranAD 等数值对照。

### 2.3 TSAD-Agents 的四个角色

v2 第 6 节针对点与区间异常构建 TSAD-Agents；作者说明变量级异常已有较好的直接视觉效果。框架依据能力实验设计，支持数值/图像模态切换，四个角色共享动态记忆。

| 角色 | 原文职责 | 对本项目的直接影响 |
| --- | --- | --- |
| Scanning | Chain-of-Scanning 先判断 regular/irregular，再判断 point/range，得到四种组合 | 数据状态扫描和异常粒度推断已有明确先例 |
| Planning | point 与 irr-point 采用传统检测、混合模态复核；range 与 irr-range 采用图像检测和复核 | 可直接借鉴以粒度为线索的工具与模态选择 |
| Detection | 从工具池调用传统或 MLLM 检测器，执行计划 | 将数值工具和视觉工具交给 Agent 并非独有设计 |
| Checking | 将预测突出显示于文本/图像中，检查范围过宽或过窄并修正 | 首版用去掉复核的消融验证作用 |

第 6.3 节说明作者使用 iForest 和 Gemini 检测工具，所有 Agent 由 Gemini 驱动并使用 LangChain。表 4 比较 Point、Range、Irr-Point、Irr-Range 四类情形，包含传统算法、Prompting、LLMAD、SIGLLM、TAMA；图 9 消融 CoS、Scanning、Planning、Checking。原文报告组合提升，但本项目不能将其论文分数作为 GNSS 结果，也不能在指标协议不同的情况下直接比较数值。

### 2.4 公开代码与重建边界

2026-09-22 复查官方 `main`，仍为提交 `7158c4ff05bc6fb27bf5045a1e8129318ca4a47d`。递归文件树主要是生成、模型调用、提示和聚合脚本；检查 `src/main.py` 与 `src/prompt.py` 未见四角色编排。因此，**当前核查的公开主分支未提供可直接运行的 TSAD-Agents 实现**；这不代表已穷尽作者其他分支、私人实现或未来发布。

`generator.py` 的 `drop` 将随机选中的数据与标签一起设为 NaN。本项目必须独立保留删点前标签和可观测性。含异常类别的文件名/目录不向模型暴露。

如果后续确需按第 6 节重建四角色流程，命名为“TSAD-Agents 风格重建”，记录原文规定与工程补足的对应。论文未明确的细节只能在开发集确定，没有官方代码与完整配置时不声称逐项复现表 4。当前首版以借鉴流程为主，完整重建不是必做项。

### 2.5 本项目的领域验证重点

1. 几种代表性数值方法与视觉模型，在典型 GNSS 异常和真实噪声背景下分别表现如何。
2. 少量随机缺测和连续缺测设置，是否改变方法优势。
3. 依据这些发现采用工具选择与复核流程，是否改善领域检测效果；用固定组合、规则选择和通用 Agent 对照即可。

“先能力实验、再自适应工具组合”可以直接作为本项目研究路线，明确引用 TSAD-Agents。本项目的价值可以落在 GNSS 领域数据、配置适配与实验验证，不将通用机制原创性作为必须额外解决的问题。

## 3 TSB-AD [R2]

**核对材料**：NeurIPS 2024 正式论文全文 PDF，重点第 3–5 节；官方项目页、README 和 `TSB_AD/evaluation/metrics.py`。OpenReview 下载返回 403 后改从 NeurIPS 官方 proceedings 获取成功。

**原文事实**：整理 40 个数据集的 1,070 条序列，包含 870 条单变量和 200 条多变量，比较 40 个检测算法与 10 个指标。强调数据瑕疵、标注质量、指标偏差和一致的基准协议；在其研究场景中推荐 VUS-PR。论文单列调参集与评估集，不主张把复杂深度网络默认当作优于简单方法。

论文讨论 point adjustment 可能让随机预测获得虚高表现，并考察 affiliation 等指标的偏差。因此新实验不能只把 VisualTimeAnomaly v1 的 affiliation 或 TAMA 的 PA-F1 原样作为唯一主指标。VUS-PR 也有缓冲长度与规则采样假设，不能未经说明套到实际时间间隔不均的 GNSS 观测。

**代码核查**：`get_metrics(score, labels, ..., pred=None)` 的注释明确说明阈值相关指标在未给 `pred` 时使用 oracle threshold；返回字段中的 `Event-based-F1` 调用的是 `metric_EventF1PA`。本项目须显式使用开发/验证阶段锁定阈值得到的预测。当前采用无point adjustment的日×分量指标，不增加事件匹配指标，不能仅凭上游字段名混用口径。

**可借鉴**：适配器、超参预算、公开数据清单、统一汇总及运行成本。

**需调整**：GNSS 基准保留真实无异常长时段、标签未知区域和数据不可判定性，不因方法普遍难检就自动删去困难案例。公开通用集可以做外部 sanity check，不能代替目标场景证据。

## 4 TAMA [R3]

**核对材料**：arXiv `2411.02465v1` 第 3 节、第 4.1/4.3 节、第 5 节及 PA 讨论；官方 README、`main_cli.py`、`evaluation.py`。

**原文事实**：流程包括 Multimodal Reference Learning、Multimodal Analyzing、Multi-scaled Self-reflection。正常参考图帮助建立正常形态描述；分析输出异常区间、类型、置信度与解释；对已检出异常的窗口使用放大图复核。重叠窗口通过位置映射和置信分数聚合生成最终输出。仓库包含图像/文本数据构建、两种 CLI、评价和消融脚本。

原文主表包括 point-adjusted F1，并讨论 PA 可能高估结果及 PAT 实验。因而不能把论文中较高 PA-F1 直接当作 GNSS 方案的可达分数，也不能将其与未使用 PA 的方法混表。

**代码核查**：`find_normal_reference` 优先从训练图像目录选参考；训练目录不存在时可从 `structure_info` 指定的 `test/image` 位置获取正常参考。这个实现细节不等于原论文所有实验都泄漏，但在本项目严格独立测试协议下不能照搬该 fallback。所有正常参考必须在切分时锁定来源。

**可借鉴**：图像与文本配对比较、0/1/3 参考样本消融、局部放大、结构化区间输出。

**本项目采用方式**：借鉴局部图复核，通过 LMA 去掉复核的消融检查其作用。首版不要求完整复现 TAMA 或开展全部 few-shot/模态消融。重复看同一幅图属于自检，不自动称为独立证据。

## 5 支撑工作

| 来源 | 本次核对深度 | 可用于本项目 | 不宜直接沿用 |
| --- | --- | --- | --- |
| AnomLLM [R4] | 论文摘要/任务与表示章节、官方 README | 文本/视觉表示比较；合成数据；online/batch API；统一 `result_agg.py` | 旧模型列表、凭据文件方式及不同任务的结果 |
| TimeEval [R5] | 官方 README | algorithm adapter、dataset manager、预处理/主运行/后处理计时、结果追踪 | 整套框架不是必需；README 说明 Windows 需考虑 WSL，Docker 算法另有依赖 |
| TODS [R6] | 官方 README 与论文元信息 | point/pattern/system 粒度划分、检测器与预处理模块化 | 厚重 AutoML 依赖不宜作为首版必需 |
| LLM-TSAD [R7] | 官方 README | 数值文本加图像的近期对照；AnomLLM/TSB-AD-U 批处理组织 | 本次未获取关联论文全文，不据此声称性能或完整机制 |
| TSB-AutoAD [R8] | 官方 README、仓库目录与引用信息 | 选择/集成/生成的对照视角；随机、朴素集成与数据驱动选择 | 本次未精读论文全文，不把仓库摘要的结论推广到 GNSS |

TSB-AutoAD 官方摘要描述其覆盖 20 个方法及 70 个变体，并指出自动方案相对随机选择并不总有显著优势，历史数据驱动方法也面临分布外退化。此处作为设计强对照的动机，具体数字和表格在论文写作阶段应进一步核对正式论文。

## 6 已核实来源

| 编号 | 工作与一手来源 |
| --- | --- |
| R1 | Xu et al. *Can Multimodal LLMs Perform Time Series Anomaly Detection?* WWW’26（arXiv 标注）。[论文 v2](https://arxiv.org/html/2502.17812v2) · [版本元信息](https://arxiv.org/abs/2502.17812v2) · [旧版 v1，仅供版本核对](https://arxiv.org/html/2502.17812v1) · [仓库](https://github.com/mllm-ts/VisualTimeAnomaly) |
| R2 | Liu and Paparrizos. *The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark.* NeurIPS 2024. [正式论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html) · [仓库](https://github.com/TheDatumOrg/TSB-AD) · [项目页](https://thedatumorg.github.io/TSB-AD/) |
| R3 | Zhuang et al. *See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers.* [论文 v1](https://arxiv.org/html/2411.02465v1) · [仓库](https://github.com/ChongKaKam/TAMA) |
| R4 | Zhou and Yu. *Can LLMs Understand Time Series Anomalies?* [论文 v1](https://arxiv.org/html/2410.05440v1) · [仓库](https://github.com/rose-stl-lab/anomllm) |
| R5 | TimeEval. [官方仓库与论文引用信息](https://github.com/TimeEval/TimeEval) · [算法仓库](https://github.com/TimeEval/TimeEval-algorithms) |
| R6 | Lai et al. *TODS: An Automated Time Series Outlier Detection System.* [论文](https://arxiv.org/abs/2009.09822) · [官方仓库](https://github.com/datamllab/tods) |
| R7 | *Delving into Large Language Models for Effective Time-Series Anomaly Detection.* [用户指定仓库](https://github.com/junwoopark92/LLM-TSAD)；本轮仅确认仓库信息 |
| R8 | Liu, Lee and Paparrizos. *TSB-AutoAD: Towards Automated Solutions for Time-Series Anomaly Detection.* PVLDB 2025，卷期页码按官方 README 为 18(11):4364–4379。[官方仓库](https://github.com/TheDatumOrg/TSB-AutoAD) |

## 7 代码核查版本

以下是调研时读取的具体提交，后续实现应另做版本与许可确认。复制代码必须保留许可和归属，不直接运行上游仓库中的凭据或远程实验配置。

| 仓库 | 核查 commit | 关键位置 |
| --- | --- | --- |
| VisualTimeAnomaly | `7158c4ff05bc6fb27bf5045a1e8129318ca4a47d` | [删点实现](https://github.com/mllm-ts/VisualTimeAnomaly/blob/7158c4ff05bc6fb27bf5045a1e8129318ca4a47d/src/generator.py#L174)；`src/main.py` |
| TSB-AD | `6beac72e11d1155ade40870492c00d0d1cfdcaaf` | [评价入口](https://github.com/TheDatumOrg/TSB-AD/blob/6beac72e11d1155ade40870492c00d0d1cfdcaaf/TSB_AD/evaluation/metrics.py#L3) |
| TAMA | `db99a75a83d0ef05caf7ff0d199b75b594e98584` | [正常参考查找](https://github.com/ChongKaKam/TAMA/blob/db99a75a83d0ef05caf7ff0d199b75b594e98584/main_cli.py#L110)；`evaluation.py` |
| TSB-AutoAD | `88b00300cc92faa7454b94bf0929cb0fb164253b` | `README.md`；方法目录 |

## 8 未核实事项

VisualTimeAnomaly 与 TSAD-Agents 的关系已通过 v2 第 6 节确认。公开 Agent 实现与若干复现配置缺失已记录，但不阻塞当前以方法借鉴为主的领域应用方案。

本项目后续已完成平台采集与模型图像接口验证，最新实施状态见[项目状态](project-status.md)。本文的论文事实不等于GNSS验证结果；注入、窗口与调用参数以本项目冻结协议为准。未精读的补充论文、供应商底层模型版本及上游未公开实现仍保留核查限制。
