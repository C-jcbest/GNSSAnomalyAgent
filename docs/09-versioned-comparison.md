# 多版本方法、评价视图与提示语义实验

> 2026-09-27。用户要求保留历史方法、精确/容差及区间多指标结果，并实际测试明确活动语义后的视觉提示。本轮协议在调用前写入，现已完成 12 次预检与 600 次正式调用；结果见第 5 节。[冻结记录](../configs/p7a-frozen.json)包含配置、源码、预测和指标哈希。

## 1. 方法版本与评价视图区分

检测器输出预测；评价视图只改变如何测量这份预测的质量。Point ±3 日不是一个新 detector，Theil–Sen 的 Affiliation 与 IoU 也不是两个不同算法。为论文可追溯性，分别记录 method_id、prediction SHA256、metric_protocol、数据版本、源码/配置/提示哈希。

保留 P5 全部四方法（SR、PELT、Matrix Profile、Rolling Theil–Sen）、P6 原提示 `visual-v1`，新增 `visual-semantics-v2`。历史预测、report、原始响应和配置不覆盖。P5 选定的方法不改变，也不按新指标重选已有基线。

## 2. `comparison-views-v1` 的评价口径

### Point

- **精确日**：保留原 P4 全案例×轴的日期集合微 Precision/Recall/F1。
- **±3 日**：逐 case×axis，GT 为 Point 起点集合，预测日期去重；连接绝对日期差不超过 3 日的边，求最大数量一对一匹配。每个预测/GT 最多用一次，未匹配预测为 FP，未匹配 GT 为 FN，再全局微汇总。不膨胀真值掩码、不做 point adjustment。
- ±1、±7 日只作为完整的预定义敏感性附表，不按最佳分数选择容差。
- 容差只影响匹配，负轴定义与 FAR 不变。任务没有阳性的分组 P/R/F1 显示 N/A，仍报告预测量和 FAR。

### Range

- **Affiliation**：保留 P4 的 GT 正例 case×axis 宏 P/R/F1，F1 是逐轴 F1 的平均，不由宏 P/R 再计算。
- **逐日 IoU**：每轴 GT 活动日期集合 G、预测日期集合 P，计算 `|G∩P|/|G∪P|`，同一固定 GT 正轴上宏平均。闭区间包含两个端点；同轴重叠预测按集合去重。
- 补充相同正轴上的逐日 Precision/Recall/F1，以及全网格日级 TP/FP/FN 微指标，后者会惩罚负轴上的预测。两者分母不同，列名明确区分。
- GT 正轴上的失败/空预测各指标记 0；没有 Range GT 的分组 Affiliation/正轴 IoU 记 N/A，不能用大量空正常轴把 IoU 抬高。

### 共同内容

- 保留原任务负轴 FAR（成功负轴中至少报一次的比例）及其计数/分母、全案例执行成功率；另外报告 30 Normal 的成功轴 FAR、案例失败率。
- 按 Normal/Single/Multi、Single 形态、Multi scenario、N/E/U 做事后分组；这些标签仅在评价进程读取，不进入提示或分支选择。
- 导出逐案例指标、总体和分组 JSON/CSV、论文表格 Markdown；保存预测哈希和引用来源。所有运行产物仍默认不入 Git。
- 新旧视觉在共同成功案例上另做成对诊断，以分离输出失败的影响；正式主表始终包括所有 300 例，不用共同成功子集替代主结果。
- 单次请求不能估计模型重复运行的方差，也不把案例间波动当重复实验标准差。此次是已看过的开发集结果，不能称独立测试或因果证明。

这套多视角报告借鉴 [VisualTimeAnomaly](https://arxiv.org/html/2502.17812v2) 分任务实验、[Local Evaluation](https://ahstat.github.io/images/2022-anomaly1-kdd-paper.pdf) 的距离指标，以及 [Navigating the metric maze](https://link.springer.com/article/10.1007/s10618-023-00988-8) 对评价偏好差异的讨论。±3 日是一项本项目的预定定位容忍度，不冒称上述论文的统一标准。P4 主协议保留，新视图单独版本化。

## 3. 本轮实际实验：只更改提示

本条按最新用户请求收束并替代 [P7 草案](08-p7-design.md)中“先 schema 两条件×60 例”的执行顺序：先对同一 300 例做 **prompt-only 全量测试**。schema-only 条件和候选复核仍未实施。

- 方法名：`visual-semantics-v2`；配置：`configs/p7a-visual-semantics.json`；目录：`runs/p7a/visual-semantics-v2/`。
- 模型仍为 `qwen3.8-flash`，thinking=false、temperature=0、max_tokens=512、4 workers；JSON Object 模式、严格原解析器和全部传输参数保持 P6 原样。每例 Point/Range 各一次，无纠错重试，失败计入。
- 复用 P6 原 PNG 字节，并对照当时请求记录逐张核对哈希。输入检查不读取 truth，评分单独执行。
- 提示只补充：Point 为孤立日或持续新水平的突变首日；Range 为持续演化的活动区间或暂时偏离到恢复前的完整区间；稳定残留偏移不延伸 Range；暂时区间边缘不另报 Point；普通噪声、平滑周期不自行构成目标；各轴可以为空、异常不要求跨轴共同出现。
- 提示不提供 GT、候选、数值摘要、类型标签、固定注入时长/幅值、位置安全区或事件数量。
- 在 Pilot 外沿用六个人工工程序列做 12 次预检，只检查结构/模型/传输；通过后执行 600 次正式请求。保留运行起止时间、响应型号、usage、失败及原文；服务未提供实际金额时 cost=null。
- 完成后固定归档，不按得分修改提示或补跑失败案例。新旧调用时间不同，结果差异可能含服务漂移与采样影响；若未来需要严格归因，应新增同期原提示对照，而不能重命名旧响应。

## 4. 运行入口

```powershell
uv run gnss-sim visual-semantics --phase preflight --env-file "C:\path\to\.env"
uv run gnss-sim visual-semantics --phase run --env-file "C:\path\to\.env"
uv run gnss-sim visual-semantics --phase evaluate
uv run python scripts/export_comparison.py
```

正式调用有独占锁和逐请求日志。已开始但未留存响应的请求在恢复时记失败，不重复提交；已经完成的运行只返回记录。新实验须用新的方法/配置/目录，不能拿本入口覆盖历史运行。

## 5. 全量开发 Pilot 结果

### Point

| 方法 | 精确 P | 精确 R | 精确 F1 | ±3 日 P | ±3 日 R | ±3 日 F1 | 负轴 FAR | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SR | 0.4490 | 0.7156 | 0.5518 | 0.4745 | 0.7562 | 0.5831 | 0.2101 | 1.0000 |
| PELT | 0.4194 | 0.2054 | 0.2758 | 0.4424 | 0.2167 | 0.2909 | 0.1458 | 1.0000 |
| Visual v1 原提示 | 0.0339 | 0.1061 | 0.0514 | 0.1993 | 0.6230 | 0.3020 | 0.6213 | 0.9600 |
| Visual v2 明确语义 | 0.0665 | 0.1129 | 0.0837 | 0.3790 | 0.6433 | 0.4770 | 0.3077 | 0.9900 |

新版的主要变化是减少误报：精确 TP 从 47 到 50，FP 从 1338 到 702；精确定位仍明显弱于 SR。±3 日指标反映大致定位能力，不能用它替代精确日结果宣称解决了定位问题。

### Range

| 方法 | Aff P | Aff R | Aff F1 | 正轴平均逐日 IoU | 全轴逐日微 F1 | 负轴 FAR | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matrix Profile | 0.0939 | 0.0954 | 0.0939 | 0.0011 | 0.0006 | 0.0499 | 1.0000 |
| Rolling Theil–Sen | 0.9284 | 0.9449 | 0.9345 | 0.3398 | 0.4886 | 0.1954 | 1.0000 |
| Visual v1 原提示 | 0.7227 | 0.8204 | 0.7573 | 0.3719 | 0.2036 | 0.5506 | 0.9100 |
| Visual v2 明确语义 | 0.8526 | 0.9009 | 0.8730 | 0.5400 | 0.4102 | 0.2446 | 0.9367 |

新版视觉在 GT 正轴的 IoU 高于 Theil–Sen，但 Affiliation、全轴微 F1、FAR 和成功率仍各有差异，不能写成视觉全面优于数值。

### 对提示变化的解释与反例

- Single Step 的合法 Range 案例两版均为 23。事件轴上的 Range 报警由 23 降至 16；延伸到第 364 日的报警由 21 降至 1。残留尾部减少，跨任务误报仍未消失。
- Single Slow Trend 的 IoU 从 0.4775 到 0.6293，Acceleration 从 0.3499 到 0.6230，Transient Shift 从 0.6562 到 0.6524。改善集中在持续演化边界，不能说所有形态都有改善。
- Normal 的成功轴 FAR：Point 从 73/81（0.9012）到 37/90（0.4111）；Range 从 48/63（0.7619）到 47/81（0.5802）。正常组报警仍然偏多，且新旧有效分母不同。Range 正常轴总预测日数从 5957 到 6662，说明不能仅凭 FAR 下降推断所有报警负担都降低。
- 共同成功的 285 个 Point 案例中，精确 F1 为 0.0510→0.0840、±3 日 F1 为 0.3006→0.4816；共同成功的 256 个 Range 案例中，Aff F1 为 0.8179→0.9170、IoU 为 0.4045→0.5656。这个辅助对照支持改善不完全由格式成功率变化造成；它不替代全 300 例主表，也不能消除调用时间及模型随机性影响。

### 失败与成本

Point 成功 297/300，2 条返回 JSON 数组、1 条 ReadTimeout；Range 成功 281/300，19 条返回 JSON 数组。所有失败保持空预测计分，没有重试或修复。JSON Object 只约束 JSON，并不保证 N/E/U 结构，因此结构可靠性仍是独立的后续问题。

正式调用耗时 269.410 秒（图片复用与输入校验在该计时前完成），已记录 599 次请求 usage，共 **1,441,389 tokens**；一次超时没有 usage，实际消耗可能更多。预检用量另存在预检原始记录，不包含在此总数。API 没有实际金额，cost=null。所有返回模型标识均为 `qwen3.8-flash`，不假定服务底层权重固定。

## 6. 论文材料与复现

- `configs/comparison-views-v1.json`：四个数值方法、两版视觉的预测路径及评价视图注册表；两版视觉各有 Point/Range，合计八行方法×任务。
- `runs/comparison-views-v1/summary.json`：总体、多指标、输入/源码/预测哈希、usage、共同成功成对诊断。
- `overall.csv`、`grouped.csv`、`per-case.csv` 及对应 JSON：主表、Normal/Single/Multi、各形态/场景/轴和逐案例分析。无阳性分组为 N/A，保留误报和失败统计。
- `paper-tables.md`：两张可直接整理进论文的表；`comparison.png` 与可编辑文字的 `comparison.svg`：Point 容差敏感性及 Range 双指标图。±1/7 日完整保留，不按最佳结果选主容差。
- `runs/p7a/visual-semantics-v2/`：每次请求日志、原始响应、预测、报告、运行记录、源码/config/依赖锁快照 ZIP 及 SHA256。凭据不进入快照或 Git；所有原始产物仅存本地。

296 项 pytest、Ruff 通过；新评价视图逐项重现原 P4 的全部八份主报告。P6 配置、源码和预测哈希保持冻结值。本轮到此归档，尚未执行 schema 对照、数值视觉复核、Agent 或独立确认集。
