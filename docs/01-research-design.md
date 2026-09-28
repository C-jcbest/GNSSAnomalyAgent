# 研究问题与实验路线

> 版本：2026-09-27 · `event-v6`、P4、[P5](06-p5-numerical-baselines.md)、[P6](07-p6-visual-baseline.md)及[P7a 提示语义与多评价视图](09-versioned-comparison.md)均已实现。当前只有合成开发 Pilot 上的比较结果，尚无独立测试结果。

## 摘要

本实验以合成的日尺度 N（北）、E（东）、U（高）三轴坐标为共同输入，逐步研究异常信号的生成、定位与评价。模拟器保存已知成分；后续检测器只能读取观测输入，并分别报告异常日期或活动区间及其分量，不要求预测异常类型。曲线形态本身不能裁决仪器故障、真实地表形变或滑坡成因。

P1 的职责是提供一个**固定、可复现的正常背景**，不是研究贡献或万能仿真器。每例恰有 365 个日期，覆盖 2025-01-01 至 2025-12-31。网页临时批次的公开生成控制项为案例类型（含 P2 混合与 P3 场景组合）、数据集 seed 和案例数；事件幅值、时长及位置不开放。P4 固定 Pilot 使用专门入口。改变背景模型、噪声类别或窗口长度都要另立实验版本。

## 研究对象与名词

| 名词 | 本实验的含义 | 边界 |
| --- | --- | --- |
| 合成监测点 | 固定参考坐标 $P_0=(0,0,0)$ mm 的三轴对象 | 不对应真实站点 |
| 案例 | 一个完整 365 日的 N/E/U 观测窗口 | P2 至多一个事件；P3 场景例 2～6 个事件 |
| 数据集 | 一次请求生成的一组案例 | 请求含 seed 与数量，案例 seed 分层派生 |
| 正常背景 $B_t$ | annual 与 semiannual 正弦项之和 | 是生成机制，不等于现场正常性认证 |
| 测量噪声 $\epsilon_t$ | 独立白噪声 | P1 没有 AR(1) 或 flicker noise |
| 观测坐标 $O_t$ | 后续检测器可看到的值 | P1 为 $P_0+B_t+\epsilon_t$ |
| 真值 | 模拟器保存的背景、噪声、相位、成分 seed 和事件 | 教学与评价可读；检测输入不得读取 |
| 异常形态 | Spike、Step、Slow Trend 等时间结构 | 形态与注入来源分开记录 |

这里的“日尺度”是模拟器的采样网格，不是真实 GNSS 日解，也没有北京时间 15:00 的取样含义。N/E/U 和所有偏移统一使用毫米；H 与 R3D 表示相对固定参考点的模长，不是累计路径长度。

## 为什么固定这个背景

GNSS 时间序列研究常把 annual 和 semiannual 项作为周期信号，参见 [Khazraei & Amiri-Simkooei, *Geophysical Journal International*, 2020](https://academic.oup.com/gji/article/224/1/257/5911580)。当前 annual 1.0/1.0/1.5 mm、semiannual 0.25/0.25/0.5 mm 和白噪声标准差 0.5/0.5/1.0 mm（均按 N/E/U）是本实验为异常可辨识性选取的 **controlled benchmark setting**，并非直接照搬该论文的参数表。

引用用于说明**周期项结构**，这组固定数值不代表任何具体现场的精度。参考研究还包含线性速度、offset 与 flicker noise；当前基准不采用它们。尤其不加入 secular deformation，避免正常组先含有与后续 Slow Trend 混淆的线性变化。若后续要研究相关噪声或不同时间窗，应建立单独 robustness 条件并留出可比较的版本。

## 分阶段流程

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| P1 | 365 日固定正常背景、输入与真值隔离、可复现生成及交互浏览 | 已纳入 `event-v6` |
| P2 | 五种单轴单事件注入、typed truth、独立贡献与交互核查 | 已实现；公式沿用至 `event-v6` |
| P3 | 六场景多事件叠加、同轴/跨轴组合、逐事件贡献 | 已实现；不叠加 Slow Trend 与 Acceleration |
| P4 | 300 例固定开发 Pilot、Point/Range 分任务真值与评价器 | 已实现；见 [P4 协议](05-p4-pilot-evaluator.md) |
| P5 | 四个数值基线、Normal 阈值校准、冻结 Point/Range 各一项 | [已验收](06-p5-numerical-baselines.md)：Point SR、Range Rolling Theil–Sen |
| P6 | N/E/U 三联图的独立零样本视觉 Point/Range 基线 | [已验收](07-p6-visual-baseline.md)：`qwen3.8-flash`，600 次正式请求 |
| P7 | 先对齐视觉契约，再验证数值与独立视觉候选的固定复核 | [P7a 语义对照已验收](09-versioned-comparison.md)；[P7b 候选复核已完成但未通过继续 Gate](12-p7b-candidate-review.md)；schema 对照未实施 |
| P8 | 失败分析后优化视觉 Range 定位 | [P8a 全局/局部视觉对照](15-p8a-visual-range-context.md)60 例未通过 Gate；用户随后指定的完整 Pilot300 测试也已完成，IoU 改善但主 F1 未超过单轮 |
| P9 | 在互补性证据支持下设计轻量 Agent 与独立测试 | 未实现 |

P1～P4 统一采用 365 日案例。Pilot 是开发验证，不是独立最终测试。P1 的正常曲线不能用来推论异常检测能力；合成注入实验也不能直接验证真实滑坡预警。

用户已将 P7a 执行顺序收束为同一 300 例的 prompt-only 语义对照，见[当前协议](09-versioned-comparison.md)。历史方法保留，新增 Point ±3 日与 Range 逐日 IoU 等补充视图；P4 主指标不变。[固定候选复核](12-p7b-candidate-review.md)已在 60 例开发子集运行，C 相对 U 主 F1 未提升，按 Gate 停止推进本版确认；保留独立基线和负结果。

[全程审查](10-experiment-audit.md)已核查数据、冻结源码、八份报告和历史调用；随后已补齐输入隔离及冻结参数数值入口，并完成 N/V/并集/固定复核四条件开发比较。直接并集存在明显误报代价，SR 对 Step 的覆盖也有限，不能预设组合或 Agent 优势。候选复核未通过主指标 Gate，独立确认数据和新结构化输出实验未启动；后续决策以 P7b 结果为准。

## 阅读路径

2026-09-28 已完成 [P7b 错误归因](13-p7b-error-analysis.md)，随后用户授权 [P8a 视觉 Range 对照](15-p8a-visual-range-context.md)。顺序调整为视觉开发优化、必要时比较固定协作与轻量 Agent、最终冻结后独立确认。[独立确认](14-independent-confirmation-protocol.md)仍是草案，确认集未生成；其最终方法表待开发决策后登记。

首次阅读先看[实验方法手册](11-experimental-handbook.md)：以论文结构统一解释术语、任务、数据、方法、指标和有效性边界。P7b 的四条件、60 例分层开发筛查、候选 ID/预算/失败规则与独立确认建议见手册第 8～10 节；已执行开发筛查，结果见 [P7b](12-p7b-candidate-review.md)；两个任务主 F1 均下降，未启动独立确认。

先看 [P1 正常序列模型](02-p1-normal-model.md) 理解背景，再看 [P2 事件与 P3 场景协议](04-planned-methods.md) 理解五种事件与六种组合；[数据契约与可复现流程](03-data-and-reproducibility.md) 说明临时批次，[P4 固定 Pilot 与 Point/Range 评价协议](05-p4-pilot-evaluator.md) 说明正式开发集和评分，[P5 数值基线](06-p5-numerical-baselines.md) 与 [P6 纯视觉基线](07-p6-visual-baseline.md) 记录两种表示的开发结果。网页的真值视图只用于生成检查；检测方法只能使用案例输入。
