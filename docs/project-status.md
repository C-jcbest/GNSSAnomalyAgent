# 项目状态与关键决定

## 2026-09-26 · P6 纯视觉基线实施前设计

- 类型：设计决定、文档同步。P6 固定为 `pilot-v1` 上独立的 Point/Range 零样本图像基线：从 `CaseInput.displacement_mm` 渲染 N/E/U 三联图，每任务每例一次视觉模型请求，严格解析为现有 P4 结果模型，仍用原评价器计分。旧文档中的 NEUH 四联图已收束为 NEU；不得加入 H、数值摘要、P5 候选、few-shot 或 Agent。正式设计见 [P6 协议](07-p6-visual-baseline.md)。
- 真值隔离：P6 推理阶段只核对 manifest 指定的输入哈希，不调用会读取 `truth.json` 的 `verify_pilot()`；两任务各生成 300 行结果后单独评价。超时、API/解析失败保留失败行，原始模型响应另存且不入 Git。先在 Pilot 外做图像和 JSON 工程预检，随后冻结图像、提示、模型与运行配置。
- 状态与待决：当前只完成协议和网页文档导航同步，没有 renderer、runner、模型端点核验、正式调用或 P6 分数。首选 Qwen2.5-VL-7B-Instruct 的实际 provider、模型修订、图像缩放和可用请求参数仍须实施前核实。P5 开发结果保持不变，不据此推断 P6 或独立测试表现。

## 2026-09-26 · P5 四个数值基线运行与方法冻结

- 类型：实现、验收。`gnss-sim numerical --method` 支持 SR、PELT、Matrix Profile、Rolling Theil–Sen；只读取固定 Pilot input，30 个 Normal 的逐轴最大分数校准三项阈值，PELT 从预定六个 beta 候选中按 Normal 误报选取，四项分别以现行 P4 evaluator 评分。300 例×四方法均执行成功；预测、report、run 记录在忽略入 Git 的 `runs/p5/`，四方法参数及选中方法另存[冻结配置](../configs/p5-frozen.json)。没有数值 ensemble、视觉模型或 Agent。
- 开发结果：Point SR 的 P/R/F1/FAR 为 0.4490/0.7156/0.5518/0.2101，PELT 为 0.4194/0.2054/0.2758/0.1458；Range MP 的 Affiliation P/R/F1/FAR 为 0.0939/0.0954/0.0939/0.0499，Rolling Theil–Sen 为 0.9284/0.9449/0.9345/0.1954。按预定 F1 优先规则，冻结 Point SR 和 Range Rolling Theil–Sen。PELT 三轴均回退最大 `β=32`，N/E/U 的 Normal 校准误报数为 3/2/0，没有为追求结果扩大候选集。完整口径与时间见 [P5 协议](06-p5-numerical-baselines.md)。同一 Normal 既用于校准又用于开发 FAR，所有结果仅说明开发 Pilot，不能推断独立泛化或现场预警。
- 验收：269 项 pytest、Ruff、`uv lock --check` 通过；四份 JSONL 各 300 条且零解析错误，SR 重跑的预测 SHA256 相同。`pilot-v1` manifest/summary SHA256 仍为 `a2e43db3493f86b36d1b962126f70f462b2ee3f4bf711bdbd84b078d43c10e33` / `a88b4ca674fc3e122f48ba798d7898af2016e02ad4e24e6328f405c62a369007`，逐例输入与真值通过冻结校验。P5 在此结束；P6 尚未开始。

## 2026-09-26 · P5 数值基线设计冻结（实施前记录）

- 类型：设计决定。P5 限定为四个独立数值基线：Point 的 SR、PELT；Range 的 Matrix Profile、Rolling Theil–Sen。P4 `pilot-v1` 数据、`event-v6` 生成器和 Point/Range evaluator 不变；不加数值 ensemble、视觉模型或 Agent。实施协议见 [P5 设计](06-p5-numerical-baselines.md)。
- 参数：MP 与 Theil–Sen 均用 30 日窗口，分数映射到右中心日；SR 的频谱平滑宽度 3；PELT 固定 L2、`min_size=3`、`jump=1`，断点直接映射新片段首日。SR/MP/Theil–Sen 的三轴阈值及 PELT 的三轴 penalty 只按 30 个 Normal 输入校准，不按异常 GT 搜索 F1。Normal 同时进入开发评价，FAR 不能当独立数据误报保证。
- 后续验收：四方法各跑固定 300 例、按 P4 原评分输出两张表，要求输出合法、失败不删例、Pilot 哈希不变、同配置预测字节稳定；开发集按预定 F1/FAR 次序冻结每任务一个方法。当前只有文档设计，没有安装依赖、数值实现、运行结果或论文性能结论。

## 2026-09-26 · P4 评价器按 VisualTimeAnomaly 路线破坏性精简

- 类型：需求修正、决定。用户将 P4 从单一 event matching 改为 Point 与 Range 两项任务；2026-09-25 条目中的一对一匹配、Spike/Step 容差、tIoU≥0.5、Onset/End MAE、CCR/MCR、派生 active/effect 视图及旧 `DetectionResult.events[]` 现为历史口径，不再用于当前预测、评分或报告。`pilot-v1` 的 300 例、seed、manifest/summary 哈希、`CaseTruth.events[]`、`event-v6` 生成器和 P1～P3 公式保持原样。正式口径见 [P4 协议](05-p4-pilot-evaluator.md)。
- 当前实现：`PointResult`/`RangeResult` 只含 case_id、method、status 与逐轴日期/区间；Point 在全 Pilot 日×轴二值网格上做精确日微 P/R/F1，Range 对有区间 GT 的 case×axis 用上游 Affiliation 实现做宏 P/R/F1。两项任务分报成功负轴 FAR 与执行成功率；失败正轴按空预测评分，失败负轴不充作干净样本。Affiliation 来自 `ahstat/affiliation-metrics-py` 的 commit `8d8449858096bbade6a6e70848d05c9cc9b846fe`，通过依赖与锁文件固定。旧 JSONL/report 不兼容，无迁移层；P5 前仍不运行检测器或模型。
- 验收：264 项 pytest、Ruff、`uv lock --check`、Point/Range CLI 冒烟测试通过。`pilot-v1` 再次验证了全部 300 例，manifest/summary SHA256 分别保持 `a2e43db3493f86b36d1b962126f70f462b2ee3f4bf711bdbd84b078d43c10e33` 与 `a88b4ca674fc3e122f48ba798d7898af2016e02ad4e24e6328f405c62a369007`；没有重生成或覆盖曲线/真值。测试覆盖精确点匹配、闭区间向量、原始 Affiliation 调用、宏/微汇总、负轴 FAR 与失败保留。
- 文档同步：README、研究路线、P2/P3 协议、网页文档目录和协作约定统一指向当前 Point/Range 预测与评分契约；旧 P4 评分细节只保留在下方注明失效的历史验收记录中。生成器和冻结数据未改。

以下条目按发生日期保留当时的决定与验收；其中标明为历史口径的预测契约和指标不能用于现行实验。

## 2026-09-25 · P4 固定 Pilot 与事件评价器验收（历史评分口径，已由 2026-09-26 替代）

- 类型：决定、状态。`event-v6` 的 P1～P3 背景、幅值、持续时间、场景和生成公式未改。`uv run gnss-sim pilot --seed 20260925` 固定本地 `data/pilots/pilot-v1/`：30 Normal、120 Single、150 Multi；五种 Single 各 24，每种 N/E/U 各 8、每 type×axis 正负各 4；六场景各 25。分层只读 seed 派生的轴/符号元数据，不按观测曲线筛选。manifest 保存源码及案例文件 SHA256，重复调用只校验。唯一真值仍是各例 `events[]`，活动/影响视图按需派生。
- 分布检查：Multi 每例 2/3/4/5/6 事件分别为 22/42/58/27/1；同轴 12、跨轴 138。六事件较少是当前场景模板自然抽样结果，未为使直方图均匀而重抽。事件总数：Spike 302、Step 141、Transient Shift 97、Slow Trend 61、Acceleration 62。生成器/真值 profile、贡献聚合与观测等式均逐例核查。
- 当时的评价：预测 `type` 可省略且不参与匹配；点事件 Spike/Step 为 ±1/±3 日，持续事件统一闭区间 tIoU≥0.5，严格单轴、一对一，先最大化 TP 再最大化时间质量。输出事件微 P/R/F1、Onset MAE、区间 IoU/End MAE、Normal FAR 与 FP/Normal、Multi MCR/CCR、执行成功和 Normal 失败率。方法失败或缺少结果保留 300 例分母。此评分规则已移除；现行规则见上方 2026-09-26 条目和 [P4 协议](05-p4-pilot-evaluator.md)。当时 P4 只有手工预测验收，没有数值、视觉或 Agent 检测结果。
- 验收：265 项 pytest、Ruff、前端构建通过；`pilot-v1` 再次执行校验而非重抽，CLI 评价的单条成功样例仍以 300 例为分母。手工预测样例覆盖阈值边界、重复/竞争、一对一、顺序及 ID 不变性、失败和汇总。所有结论仅为数据与评价器工程验收，不是检测性能结论。

## 2026-09-24 · P3 主图标注显示修正

- 类型：纠正、状态。P3 页面此前仅在点选时间线事件后才绘制主图标记，因此默认没有异常区域，显示/隐藏按钮也缺少可见效果。现在主图默认绘制全部已注入事件：Spike/Step 为起点竖线，Slow Trend/Acceleration/Transient Shift 为浅色时间区间；点选后突出对应事件。隐藏标注会移除主图标记及时间线，重新显示后恢复。带标注的图仍仅供研究人员核对，不作为视觉模型输入。
- 验收：在本地 `event-v6` 的 S6 五事件案例中，浏览器逐次检查默认标注、隐藏后的无标注曲线、重新显示后的标注恢复；前端构建通过。

## 2026-09-24 · P3 六场景多事件协议验收（当前 `event-v6`）

- 类型：决定、状态。P3 复用 P1 背景与 P2 五种纯事件公式、绝对毫米幅值；按六种模板生成 2～6 个事件，不把 Slow Trend 与 Acceleration 相加。长期事件至多一个、Step/Transient 至多两个、Spike 至多四个。安全区索引 60～304；多个 Spike 相隔至少 7 日、Step 至少 45 日、Transient 不重叠。S6 至少涉及两轴，S4/S5 的 Spike 可落在长期事件内。正式规则见 [P2/P3 协议](04-planned-methods.md)。
- 契约：`event-v6` / `event-*-v6` 的 truth 新增仅供检查的 `scenario_type` 与逐事件 `event_contributions`；事件按起点及固定类型优先级编号。聚合形变/伪差分别为对应事件贡献之和。typed truth 和 `DetectionResult` 不设事件数上限；后者只是结果 schema，尚无检测器或评价器。输入文件仍不含场景与真值，旧批次不迁移。网页按场景折叠案例，独立时间线支持轴/事件族筛选、点选高亮与独立贡献曲线。
- 验收：seed `20260924` 的 `all_scenarios` 生成 54 个 canonical 案例，六场景各 9 例，共 210 个事件，每例 2～6 个。自动检查 30 组 seed × 六场景的数量、间隔、位置、跨轴、事件 profile 重建、逐事件与聚合数组、逐元素观测等式，以及与 normal 的背景/噪声逐值相同；242 项 pytest、Ruff、前端构建通过。浏览器核对了生成历史、六场景分组、S6 五事件时间线、筛选、点选详情、独立贡献曲线及桌面/手机布局。该验收只证明生成与界面工作，不是异常识别或现场监测结果。P4 待后续指令。

## 2026-09-24 · P2 混合批次与默认标注（历史 `event-v5`）

- 类型：需求变更、状态。一次生成历史现在可包含 normal 与五类单轴单事件案例；`case_type=all` 固定目标比例为正常 25%、各异常 15%，至少 6 例。小批次先保每类 1 例，再按最大余数分配，平局与顺序由独立 dataset seed 命名空间决定。这是工作台检查比例，不是 P4 Pilot 配额；每例的背景、噪声和五种事件公式沿用 `event-v4`。
- 契约：升为 `event-v5` / `event-*-v5`。manifest 保存计划 `type_counts`，案例摘要保存 `case_type`；前端由摘要按类型折叠，无需为分组读取 truth。研究人员选择案例后默认从独立 truth 接口加载事件标注，可隐藏；正式检测器仍只能读取 input。新 API 不读取旧版本数据，未编写迁移层。具体字段与 seed 规则见[数据契约](03-data-and-reproducibility.md)。
- 清理：模拟工作树 `data/generated` 中 32 项旧批次与检查产物（约 127 MB）已送入 Windows 回收站，随后重建空目录；该工作树没有独立的 `runs` 或 `artifacts` 目录。清理后重新生成 20 例和网页提交的 6 例 `event-v5` 混合批次，页面生成历史仅显示这两条新批次。
- 验收：57 项 pytest、Ruff 与前端构建通过。20 例批次为 normal 5、五类事件各 3；6 例批次每类各 1。网页检查了生成、进度、混合历史、按类型折叠、默认 Spike 标线和事件时间线，以及窄屏布局。相同 seed 与案例序号在混合/单类型批次中的背景和噪声逐值一致。进度轮询遇到 Windows 短暂 manifest 文件锁时会重试。本项仅验收数据组织与研究人员界面，不产生检测结果。

## 2026-09-24 · P2 噪声与事件幅值解耦（历史 `event-v4`）

- 类型：纠正、决定。固定 annual 幅值 N/E/U 为 1.0/1.0/1.5 mm、semiannual 为 0.25/0.25/0.5 mm；白噪声标准差由 0.75/0.75/1.5 mm 降至 **0.5/0.5/1.0 mm**。五类事件保留先前的绝对幅值：Spike、Slow Trend 终值、Acceleration 终值为 N/E 4.5 mm、U 9.0 mm；Step、Transient Shift 为 N/E 3.75 mm、U 7.5 mm。未来不得用当前噪声 σ 动态换算事件幅值，或把 `sigma_multiplier` 写入 P2 真值。以上均是受控 benchmark 参数，不代表现场 GNSS 精度。
- 范围：生成器、typed truth、输入/真值/manifest 版本、网页固定协议、测试及实验文档均升为 `event-v4`；旧数据不迁移、不兼容。365 日日网格、独立相位与事件 seed、五种单轴单事件公式保持不变。正式协议见 [正常背景](02-p1-normal-model.md)、[P2 事件](04-planned-methods.md) 和 [数据契约](03-data-and-reproducibility.md)。
- 验收：55 项 pytest、Ruff 和前端构建通过。以数据集 seed `20260924` 重新生成 normal 与五类事件各 30 例，共 180 例；五类×N/E/U 的 15 个 canonical 案例逐项核对绝对幅值、起止索引/日期、事件后状态及逐元素组合等式，最大浮点残差为 $1.78\times10^{-15}$ mm。相同 case seed 的六种变体在 30 组中背景和噪声数组逐值完全一致；与同 seed 的 `event-v3` 相比，相位、背景和事件贡献逐值不变，白噪声各分量按 $2/3$ 缩放。网页核对了 Normal 观测/成分、Slow Trend 时间线与详情、Acceleration 凸形贡献。
- Normal 视觉检查：三条样例的背景年内峰峰值约 N 2.01～2.19 mm、E 2.01～2.20 mm、U 3.32～3.47 mm，周期变化仍可见；逐日随机波动也仍可见。30 条 Normal 序列每轴各有 10,920 个相邻日差，超过仅用于描述的 N/E/U 3/3/6 mm 阈值分别为 0/1/1 次，最大日差为 2.49714/3.00008/7.01191 mm。未见频繁的大跳变；零星 U 轴尖锐波动不单独视为生成错误，不继续为追求平滑下调 σ。该检查不构成检测性能或现场真实性结论。

## 2026-09-24 · 周期幅值破坏性修正

- 类型：纠正、决定。`event-v2` 的 annual 1.5/1.5/2.0 mm 与 semiannual 0.5/0.5/1.0 mm 在当前检查目标下偏强；`event-v3` 固定为 annual 1.0/1.0/1.5 mm、semiannual 0.25/0.25/0.5 mm（均按 N/E/U）。白噪声 0.75/0.75/1.5 mm、365 日网格、独立相位 seed 和五类异常参数保持不变。
- 范围：生成器版本、数据 schema、API 可读批次、网页固定协议、测试及实验文档。旧数据不迁移、不兼容；未来改背景参数必须再升版本，不能在同一版本内悄悄变更。
- 验收：49 项 pytest、Ruff 和前端构建通过。以 seed `20260924` 重新生成 normal 与五类事件各 30 例，共 180 例；选出五类×N/E/U 的 15 个 canonical 案例，逐项核对贡献的起点、终点和下一日，并在网页核对时间线。与相同 seed 的 `event-v2` 逐例比较，180 例的白噪声、两组相位、成分/事件 seed、事件 truth 和形变/伪差数组完全一致，只有正常背景变化。Normal 三例的背景年内峰峰值约为 N 2.00～2.19 mm、E 2.01～2.20 mm、U 3.32～3.47 mm；网页叠图可见周期起伏，未压过短期噪声。Slow Trend 起点为零、90 日内日增量恒定、终点后保持偏移；Acceleration 的日增量绝对值严格增大，终点后保持偏移。以上是生成器检查，不是检测性能评价。

## 2026-09-24 · P2 单轴单事件协议验收（历史 `event-v2`）

- 当时生成器为 `event-v2`，旧 `normal-v1`、`event-v1` 数据和 API 契约均不兼容，也不迁移。2025 年 365 日、每日一值，固定 $P_0=(0,0,0)$ mm；annual 幅值 N/E/U 为 1.5/1.5/2.0 mm，semiannual 为 0.5/0.5/1.0 mm，白噪声标准差为 0.75/0.75/1.5 mm。三轴两组相位由独立 seed 抽样。数值是改善异常可辨识性的受控 benchmark 设定，并非现场精度或参考论文的完整参数。
- Normal 没有事件、线性速度、AR(1) 或 flicker noise。P2 异常例恰好一个事件和一个 N/E/U 轴：Spike 为单日观测伪差 $\pm6\sigma_c$；Step 为永久形变 offset $\pm5\sigma_c$；Slow Trend 为 90 日线性累计 $\pm6\sigma_c$；Acceleration 为独立的 90 日归一化二次累计 $\pm6\sigma_c$；Transient Shift 为 14 日矩形观测伪差 $\pm5\sigma_c$。事件不早于索引 60 开始，且不晚于索引 304 结束。正式公式见 [P2 协议](04-planned-methods.md)。
- 输入和真值分别保存。事件使用 typed truth，含类型、来源、轴、起止索引/日期、持续性、幅值/终值及时长；真值另存 deformation/artifact 数组。网页初始只读 input，研究人员主动点击后才读 truth；带真值的交互图只用于检查，不是未来 Visual benchmark 输入。
- 验收：49 项 pytest、Ruff、前端构建通过。以数据集 seed `20260924` 生成 normal 和五类异常各 30 例，共 180 例；逐元素组合等式成立，六种变体相同 case seed 的 background/noise 数组逐值完全一致。每类选 E/U/N 各一例，共 15 例，人工核对了贡献在起点、终点、终点次日和年末的值，并在网页逐例核对类型、轴和 Day 区间。桌面/手机工作台、事件详情、成分切换与文档公式也已检查。
- P2 不含多事件、多轴事件、缺测、检测器、模型 API、评价指标或 Agent。下一阶段须另行指令；当前工程自检不构成检测性能或真实滑坡预警结论。
