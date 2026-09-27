# P6 纯视觉 MLLM 基线与冻结结果

> 版本：2026-09-27 · **已实施并验收**。`visual-v1` 在固定 `pilot-v1` 上完成 300 个 Point 和 300 个 Range 请求；配置与产物哈希见[冻结记录](../configs/p6-frozen.json)。以下保留执行前冻结的规则，并在末尾记录开发结果。

## 研究问题与边界

P6 只检验：**把同一批 N/E/U 观测绘成时间序列图，仅给图像与短任务提示时，一个固定 MLLM 能定位哪些异常日期和活动区间？** 输入为 P4 的 `pilot-v1` 共 300 例，不重抽样、不修改 `event-v6`、Pilot 或 P4 评价器。P5 已冻结的 Point SR 与 Range Rolling Theil–Sen 仅在评分后作对照，不进入 P6 提示或模型输入。

这借鉴 [VisualTimeAnomaly（arXiv 2502.17812v2）](https://arxiv.org/abs/2502.17812v2) 对 point/range 的分任务、零样本视觉检测路线；[AnomLLM](https://arxiv.org/abs/2410.05440) 对图像表示和显式推理提示的观察、以及 [Plots Unlock](https://arxiv.org/abs/2410.02637) 的绘图输入研究提供动机。下文的三轴渲染、严格 JSON、失败计分和 P4 汇总是**本项目协议**，不是上述论文的完整复现。P6 不评价异常类型或成因；合成 Pilot 结果不证明真实 GNSS 或滑坡预警能力。

P6 的两个任务各自独立调用同一个模型：

| 任务 | 模型看到 | 预测结构 | P4 原评分 |
| --- | --- | --- | --- |
| Point | 一张 N/E/U 图 + Point 提示 | 每轴的 0～364 日期索引 | 精确日微 Precision/Recall/F1、负轴 FAR、执行成功率 |
| Range | 同一张图 + Range 提示 | 每轴闭区间 `[start, end]` | 正轴 case×axis 宏 Affiliation Precision/Recall/F1、负轴 FAR、执行成功率 |

Point 的真值是 Spike/Step 起点，Range 的真值是其他三类事件的活动区间；**这些生成形态和分组不告知模型**。Point 提示允许描述“孤立异常日期或突变日期”这一检测目标，但不得透露模拟器的类型、参数或配额。两个任务不互相读取预测，也不合并成一个总分。

## `visual-v1` 图像协议

`src/gnss_sim/visual.py` 提供 `render_case(case: CaseInput, output: Path) -> None`。它只读取 `CaseInput.displacement_mm` 的带符号 N/E/U 三轴日值。不得接收 `CaseTruth`、`Event`、`scenario`、`case_type` 或任何预测。现有网页带真值标记的图不能作为模型输入。

| 项目 | 固定设计 |
| --- | --- |
| 格式与画布 | PNG，1800×1200 像素，白背景；固定无损编码 |
| 面板 | 3 行 1 列，顺序 N、E、U；每面板一条相同样式的实线，约 1 pt |
| 横轴 | 共同 0～364 日索引，固定范围；三个面板均显示可读索引刻度，至少标清 0 与 364 |
| 纵轴 | `Displacement (mm)`；每个面板仅按该轴**自身观测值**自动缩放并显示数值刻度 |
| 图中文字 | 只含轴名、日期索引和位移单位；无 case ID、组别、类型、事件数、注释或图例中的诊断信息 |

所有案例用同一布局、字体、刻度规则、线宽、颜色与软件依赖版本。不能依据 Normal/Single/Multi、模型反馈或观测曲线“是否明显”改变绘图设置。N/E/U 是与 P5 相同的原始信息来源，但图像会产生空间分辨率和模型内部缩放限制；因此结果比较是**相同数据的不同表示与方法**，不宣称两边拥有相同的数值精度。Point 的精确日评分尤其受图像索引可读性限制。

## 两个固定零样本提示

以下文本已逐字写入[版本化配置](../configs/p6-visual.json)并记录 SHA256。每次请求只附一张对应案例的 PNG。无需演示样例、解释、思维链、数值摘要、P5 候选或统计特征。

`point-prompt-v1`：

```text
The image shows three synchronized daily GNSS displacement time series: N, E, and U.
The x-axis is the day index from 0 to 364.
Detect isolated anomalous days or dates of abrupt changes on each axis.
Return JSON only with exactly these keys:
{"N": [day_index, ...], "E": [day_index, ...], "U": [day_index, ...]}
Use an empty list when no point anomaly is detected. Do not explain your answer.
```

`range-prompt-v1`：

```text
The image shows three synchronized daily GNSS displacement time series: N, E, and U.
The x-axis is the day index from 0 to 364.
Detect continuous anomalous time ranges on each axis.
Return JSON only with exactly these keys:
{"N": [[start, end], ...], "E": [[start, end], ...], "U": [[start, end], ...]}
Both endpoints are inclusive. Use an empty list when no anomalous range is detected.
Do not explain your answer.
```

提示不出现 Spike、Step、Slow Trend、Acceleration、Transient Shift 或其数量、位置、幅值。输出不得要求置信度或原因。Point 数组中的每个整数表示一个日期；Range 的 `start <= end` 且两端均计入区间。

## 模型、预检与冻结点

P6 主模型固定为 **`qwen3.8-flash`**，与现有 GNSS 项目的视觉模型选择一致。请求中的模型 ID 必须显式为该值，不受旧 `QWEN_MODEL` 环境变量影响。阿里云百炼的[模型说明](https://help.aliyun.com/zh/model-studio/qwen3-8-flash)确认它支持图像输入；P6 实施时仍须核对实际调用端点、响应模型 ID 和图像处理方式。若现有端点不能使用该模型，先停止正式 Pilot 运行并记录原因；更换型号需要另立协议版本，不通过试跑 Pilot 择优。

固定 `temperature=0`、`enable_thinking=false`、最大输出 512 tokens，`top_p` 不发送；发送 `vl_high_resolution_images=false`、`max_pixels=2160000`，即原 PNG 的像素数。关闭思考作为请求参数显式发送。[百炼视觉调用说明](https://help.aliyun.com/zh/model-studio/vision)列出这些图像与思考参数。`temperature=0` 不保证不同服务器或模型修订下字节级一致。运行记录包含实际端点、精确请求/响应模型 ID、可获得的模型修订/版本、请求参数、依赖版本、提示与 renderer 源码哈希、Pilot manifest/summary 哈希及逐次用量和耗时。服务没有提供稳定底层权重修订，不能假定固定。凭据不进入文档、配置或 Git。

先用 5～10 个 **Pilot 以外的人工工程序列**检查图片像素、0/364 刻度、N/E/U 顺序、API 实际图像缩放和 JSON 格式。只修坐标/传输/解析契约问题；不得用 Pilot GT、P4 分数或 P5 预测选择画法、提示或模型。预检完成后一次冻结 `visual-v1`、`renderer-v1`、两个 prompt 版本和 `model-v1`，再开始正式 300×2 任务。正式阶段每个 case×task 只发一次模型请求；超时、API 错误、空响应和非法结构直接记失败，不做内容修复或按结果重试。工程预检请求另外计数，不混入正式 600 次计划调用。

## 解析、真值隔离与运行产物

模型正文先去除首尾空白；只允许整段回答包在单个 `json` 代码围栏中。随后严格 JSON 解析并验证键恰为 N/E/U，日期是 0～364 的整数，Range 每项恰有两个整数且 `start <= end`。不能从自然语言中截取数字、补括号、改日期、删非法预测或只保留合法轴。合法正文直接构造现有 `PointResult`/`RangeResult`，`status="success"`；任何失败生成该任务案例的 `status="failed"` 和三轴空预测，并记录原因。原始回复另存，不加入 P4 schema。

推理入口仅从 manifest 取有序 `case_id` 与 `input_sha256`，读 `cases/<id>/input.json` 并逐文件核对 SHA256；`case_type`、`axis`、`sign`、`event_count` 等 manifest 字段不进入模型流程。renderer 和 runner 都不得打开 `truth.json` 或导入 `CaseTruth`。现有 `verify_pilot()` 会逐例读取真值，**不得在 P6 推理阶段调用**。输入验证失败也产生对应失败行；不跳过案例。预测全部写完并检查两任务各有 300 个不同 case ID 后，单独运行 P4 评价命令；仅此阶段读取真值并可执行完整 Pilot 校验。运行前后记录 manifest、summary 和各 input 哈希；冻结数据若发生变化则停止并标记实验无效。

运行文件存于已忽略入 Git 的 `runs/p6/qwen3.8-flash/`：`images/`、`point/raw.jsonl`、`point/predictions.jsonl`、`point/report.json`，Range 同构，另有 `run.json`。`raw.jsonl` 保存响应原文/错误、耗时、响应模型 ID、API 返回的 token 用量及请求状态；`predictions.jsonl` 每任务恰 300 行，严格是 P4 schema。`run.json` 保存配置及哈希、成功/失败和正式调用数、总耗时、汇总 token；API 未返回实际费用，记 `null`。凭据和原始响应均不入 Git。

## 报告与 P6 验收

Point 与 Range 各自产出 P4 原报告，并另列执行成功率、正式调用数、耗时及可获得的用量/费用。对照 P5 冻结的 Point SR 和 Range Rolling Theil–Sen 时仅并排展示同任务指标，不创建视觉专用分数或跨任务 overall score。可在**结果冻结后**按 Normal/Single/Multi、同轴/跨轴等做离线错误分析；它们不是新主指标，也不能反向修改 P6 配置。开发 Pilot 上的选择与对照不构成独立测试结论。

实施验收需覆盖：相同输入的 PNG 哈希/尺寸稳定、三轴顺序与索引刻度可读、无真值注释、合法/非法 JSON、越界索引与空响应、推理阶段无 `truth.json` 读取、失败案例不缺行、两份各 300 行输出可由 P4 原评价器读取、Pilot 哈希不变。完成图像与模型配置预检、600 个正式 case×task 结果、两份 P4 报告后即停止 P6。P7 才考虑数值候选、特征、复核或组合；H/R3D、few-shot、提示变体和模型比较须另立研究版本。

## 2026-09-27 开发运行与验收

人工构造的 6 条非 Pilot 序列分别执行 Point/Range，共 12 次预检请求，均返回合法 JSON，响应模型 ID 均为 `qwen3.8-flash`，没有思考内容。1800×1200 PNG 的面板顺序、纵轴单位与每面板 0/364 刻度已人工查看。随后对固定 Pilot 的 300 张图分别执行两个任务，共 **600 次正式请求**，单任务单案例各一次；两份预测各有 300 行，Pilot manifest/summary 和逐例输入哈希保持不变，评价阶段才读取真值。

| 任务 | 方法 | P | R | F1 | 负轴 FAR | 执行成功率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Point | P5 SR | 0.4490 | 0.7156 | 0.5518 | 0.2101 | 1.0000 |
| Point | P6 纯视觉 | 0.0339 | 0.1061 | 0.0514 | 0.6213 | 0.9600 |
| Range Affiliation | P5 Rolling Theil–Sen | 0.9284 | 0.9449 | 0.9345 | 0.1954 | 1.0000 |
| Range Affiliation | P6 纯视觉 | 0.7227 | 0.8204 | 0.7573 | 0.5506 | 0.9100 |

P6 Point 成功 288/300，Range 成功 273/300。39 条失败响应的 HTTP 状态均为 200，但回复主体是 JSON **数组**，不符合已冻结的 N/E/U 对象契约：Point 12 条、Range 27 条。严格解析器将其全部记为失败；没有抽取数组、改写输出或补跑。负轴 FAR 只在成功轴上计算，必须连同执行成功率阅读。P5/P6 表格只比较同一开发 Pilot 上的任务分数；P5 的 Normal 校准、图像表示精度和 P6 结构失败均影响解释，不能推广到独立测试或现场。

正式模型请求耗时 227.001 秒（4 并发）。Point 输入/输出 token 为 658200/10227，Range 为 660000/9358；合计 **1,337,785 tokens**。API 未返回实际收费金额，因此费用记 `null`，不能把未计价写成零。运行时的原始响应、逐例预测、完整 P4 报告与用量记录只在本机忽略目录；受版本控制的[冻结记录](../configs/p6-frozen.json)保存输入、配置、源码、预测和报告哈希，不提交原始数据或响应。

284 项 pytest、Ruff、`uv lock --check` 和网页构建通过；P6 在此停止。Point 表现明显低于 P5 SR，Range 也低于 P5 Rolling Theil–Sen，且视觉负轴报警更频繁。后续若分析互补性，应使用已冻结的两路预测，不能据这张开发表改变 P6 提示、画法或模型。
