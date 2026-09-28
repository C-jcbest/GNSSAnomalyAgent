# 项目状态与关键决定

## 2026-09-28 · 用户指定的 P8a 完整 Pilot300 测试完成

- 用户明确选择完整 pilot-v1，授权在原 60 例 Gate 失败后继续全集开发测试；这一指令不等于原 Gate 通过。新增 `p8a_full.py`，复用原 60 例请求/失败，新增其余 240 例，不调参、不生成独立数据、不运行 Agent。设置仍是三组 Range、qwen3.8-flash、thinking=false、8192 tokens，冻结 Theil–Sen 作为参考。
- 全集结果：V0/V1/V2 的 Affiliation F1 为 0.7788/0.7621/0.7652，正轴日 IoU 为 0.4591/0.4546/0.5284，负轴 FAR 为 0.2676/0.1964/0.2074，成功 266/259/265（各 300 分母）。Theil–Sen 为 F1 0.9345、IoU 0.3398、FAR 0.1954。局部图改善重叠但主 F1/关联召回未超过单轮；不宣称整体提升。
- 成本与失败：本次新增 660 次/2552709 tokens；全集含缓存 832 次/3245977 tokens。34 个 V0 结构失败向复核传播，V1/V2 各新增 7/1 个结构失败；所有请求正常 stop，最大回复仅 112 tokens，无截断证据。原 60 例逐行保留；报告另列新增 240 例与 Normal 诊断。
- 验证：2449 个产物哈希、832 个请求与发送图像/提示及回复重解析通过；13 项相关测试和 Ruff 通过。原始结果位于 `runs/p8a-full/`，冻结索引 `configs/p8a-full-frozen.json`；[P8a 文档第 9 节](15-p8a-visual-range-context.md)已更新，可在网页读取。此前 341 项全量测试属于原 P8a 验证，本扩展未重复全量。
- 限制：全部为已知开发 Pilot，原 60 与新增 240 并非同期，不能称独立确认；未将本次结果补写为原 60 例的预登记结论。后续另立结构可靠性和背景误报实验，不按结果重跑本轮。

## 2026-09-28 · P8a 视觉 Range 上下文对照完成

- 类型：实现、开发实验、结果与文档同步。用户授权先优化纯视觉 Range，再按证据考虑协作和 Agent；[P8a 协议与结果](15-p8a-visual-range-context.md)记录同一已知开发 60 例的 V0 单轮、V1 原图复核、V2 全局加局部图复核。三组统一 qwen3.8-flash、thinking=false、8192 tokens、JSON Object；V1/V2 共享初轮结果，提示相同，允许修订和补充范围。
- 实现：新增纯输入 `p8a.py`、排他发送前账本、单次请求与中断未知失败、预登记配置/源码/输入哈希、确定性窗口及同轴全局纵轴尺度。工程预检 6 次通过，正式请求 172 次，三条件均保留 60 行；成功 56/55/55，4 个 V0 失败向两组传播，复核各新增 1 个结构失败。无重试、无历史结果覆盖。
- 结果：V0/V1/V2 Affiliation F1 为 0.7681/0.7437/0.7420，IoU 为 0.5088/0.4875/0.5805，负轴 FAR 为 0.3182/0.2326/0.2385。局部图改善日覆盖，但主 F1、Affiliation recall 和成功率相对 V0 下降，预登记 Gate 未通过；不扩展到完整 Pilot、Agent 或独立确认。六个开发 Normal 三条件均为 13/18 轴报警，未解决背景误报。
- 诊断：V0→V2 成功配对新增 50/删除 68 正确日，新增 408/删除 1640 错误日；额外一个阳性案例结构失败另损失 45 正确日。6 条正式结构失败均正常 stop，实际最大回复 55 tokens，无 token 截断证据。全实验 178 次新调用、721383 tokens；V2 单独部署端到端 541105 tokens，约 V0 的 3.70 倍，金额未知。
- 审计与验证：521 个正式产物哈希、172 个发送提示/参数/图像与回复重解析通过。全量 341 项 pytest、Ruff、前端 TypeScript/Vite 构建通过；只存量依赖弃用与构建包体警告。冻结索引 `configs/p8a-frozen.json`，报告/逐案例/分组/日变化/CSV 和格式诊断在 `runs/p8a/`。README、方法手册、研究路线、确认草案及网页目录同步。
- 限制：全部为已知开发数据，未做独立显著性确认；V2 同时增加图像数量、局部刻度与时间放大，不能区分各因素。共同成功 54 例仅事后诊断，正式分母始终 60。下步优先研究结构约束与 Normal 误报来源，需新协议，不重复调用修补本轮成绩。

## 2026-09-28 · 后续视觉回复上限改为 8192，核验 JSON 格式

- 用户指令：后续模型最大回复 tokens=8192。新增 `configs/p8-visual-8k.json` 与 `visual_8k.py` 请求适配器，方法标识 `visual-semantics-v2-8k`，保留原 v2 提示、qwen3.8-flash、thinking=false、JSON Object；确认协议同步为新预算版本。历史 P6/P7a=512、P7b=2048 的配置、源码、结果不改。
- 核验：历史请求均带 response_format=json_object，未发送服务端 JSON Schema。P6 39 条、P7a 21 条结构失败均是合法 JSON 数组；P7a 另 1 条没有响应。P7b 107 条均为标准 JSON 对象且任务校验合法，正常结束，最大输出 62 tokens。没有发现 length 截断，不能把提高预算当作已证实的结构失败修复。
- 状态：请求适配器可发送新配置，确认数据/运行器仍待实施；旧 CLI 保持其冻结协议。无新模型调用，实际端点是否接受 8192 须在正式新实验的工程预检中验证，不能自动回退预算。详情见[确认协议 JSON 核查](14-independent-confirmation-protocol.md)。

## 2026-09-28 · P7b 离线错误归因与确认草案

- 类型：检查、诊断和方案；没有新增模型调用或生成确认数据。[错误归因](13-p7b-error-analysis.md)核对 387 个冻结产物、107 条请求/响应，精确重建 U/C 并重算八份报告，全部一致。
- 发现：33 个 Point 误删 TP 中 31 个 Spike、30 个数值独有，18 个精确日期被删后附近仍有 ±3 日预测；邻近统计不等于一对一匹配 TP。Range 419 个损失日按唯一日期去重，126 Slow/212 Acceleration/81 Transient；前两类前段损失较明显，Transient 中后段损失较明显。短候选保留率更低，不能宣称复核只删尾部或偏爱短区间；来源、长度等互相混杂。
- 工程缺口：当前数值入口未逐例捕获计算异常，也未在入口强制比对参数文件与预登记哈希；旧视觉/生成入口依赖 pilot-v1，P7b 缺发送前可恢复账本，bootstrap 未实现。正常完整的本次结果已核验，这些缺口作为新确认入口 Gate，不覆盖冻结实现或补写历史响应。
- 决策：[确认协议草案](14-independent-confirmation-protocol.md)限定独立 SR/Theil–Sen 与 Visual semantic-v2，拟新 300 例、冻结阈值、不校准确认 Normal、600 次正式视觉请求及 12 次工程预检、12 层 case 配对 bootstrap。未登记可运行版本；不恢复 P7b 组合或自动扩展 Agent。
- 材料：`scripts/analyze_p7b_errors.py` 输出 `runs/p7b-errors-v1/` 的候选/逐日/失败账本、汇总和哈希，拒绝覆盖；新增四项针对重叠去重、来源分区、inclusive 分段与误删计数的测试。两份新文档已加入 README 和网页文档目录。
- 验证：误差分析、候选复核、指标视图及文档 API 共 35 项相关测试通过，Ruff、前端 TypeScript/Vite 构建及 diff 检查通过；未改冻结推理源码、提示、参数或评分实现。

## 2026-09-27 · P7b 固定候选复核完成，继续 Gate 未通过

- 类型：实现、开发实验、结果与文档同步。执行规格及论文式结果见 [P7b](12-p7b-candidate-review.md)；`configs/p7b-registered.json` 在正式调用前登记，`configs/p7b-frozen.json` 保存源码、输入、原始响应及预测/报告哈希。没有改变 event-v6、P4、P5/P6/P7a 冻结内容。
- 实现：纯输入包只含输入、原 PNG、归档预测及非标签哈希；元数据分层在准备阶段完成，推理不读取 truth。新增冻结阈值数值入口，60 例 × 两任务与归档预测完全相同。候选精确去重、稳定 ID、保留来源且不发送模型，严格 keep_ids、64/18 批预算、一次尝试、依赖失败及整例批次失败规则已实现；所有新运行目录拒绝覆盖。
- 执行：6 Normal/30 Single/24 Multi，共 60 个已知开发案例；4 次 Pilot 外预检及 107 次正式复核结构均合法，无补救调用。Point C 成功 60/60；Range C 成功 55/60，5 个归档 V 失败完整保留。复核输入 263947、输出 1904 tokens；预检 13660 tokens，共新增 111 次请求、279511 tokens。归档上游 120 次视觉调用另记用量，未知成本不填零。
- 结果：U→C 的 Point 精确 F1 0.3512→0.2955，删除 127 FP 同时损失 33 TP；±3 日 F1 0.4167→0.5909。Range Affiliation F1 0.8498→0.7597，IoU 0.4686→0.4126；成功配对中删除 2119 FP 日、损失 419 TP 日。FAR 两任务均下降，不能据补充指标或 FAR 改善宣称主任务成功。
- 决策：两项主 F1 均未满足 C>U 的预登记继续条件，不启动独立 300 例确认、bootstrap 或 Agent；保留负结果和独立 N/V 对照。模型候选生成与复核不同期、60 例都是开发样本、Normal 比例样本小，不作泛化或模型整体能力结论。
- 验证：新增 21 项验收测试，全量 320 项 pytest 通过；107 条正式响应逐批复核并重建 115 个非依赖失败任务输出，8 份报告精确重算一致。导出逐案例/分组 JSON、论文 CSV 与 PNG/SVG；历史 300 例多版本表和本次 60 例表分别保存，避免混合排名。网页手册、协议导航和检测运行页已同步本次状态。

## 2026-09-27 · 论文式方法手册与网页阅读器

- 类型：用户要求、文档与界面实现。新增[实验方法手册](11-experimental-handbook.md)，采用摘要、研究问题、术语符号、数据、方法、指标、结果、实验规格、统计成本和有效性讨论结构；引用 VisualTimeAnomaly/TSAD-Agents、VLM4TS、Affiliation 原文、指标综述和严格评价研究。明确 Point/Range、活动与残留、micro/macro、指标分母、N/A 和失败处理；原 P4/P5/P6/P7a 冻结内容与成绩不改。
- 后续规格：用户同意推进固定候选复核方向，手册第 8～10 节给出 N/V/U/C 四条件、60 例元数据分层开发子集、固定候选 ID、保留/删除、调用预算和独立确认前置条件。新 U/C 采用严格依赖成功规则，与审查中“视觉失败回退数值”的并集诊断分开。尚未实现新运行器、冻结最终提示、生成确认数据或调用模型；本轮只完成方法说明和网页。
- 网页：默认从方法手册开始，按阅读起点/冻结协议/比较设计分组；提供文档全文搜索、二三级标题目录、滚动定位、章节地址、历史前进后退、链接复制、Markdown 下载与打印样式。移动端折叠目录，公式/表格在容器内滚动；文内 Markdown 协议链接转为站内导航。检测运行页修正过时的“尚无运行”，提供已归档结果与方法说明入口，并明确尚无运行明细浏览器。
- 验证：文档/API 相关 6 项 pytest、Ruff、TypeScript 与 Vite 构建通过；浏览器核对桌面/390px 手机布局、全文搜索、协议链接、刷新/后退、链接复制，手册零 KaTeX 错误。原稿下载已保存到本机 Downloads；打印为浏览器打印入口和专用样式，未生成独立 PDF 验收件。

## 2026-09-27 · P1～P7a 全程审查与下一阶段建议

- 类型：审查、文档纠正与待实施设计。本次无模型调用；新增 `scripts/audit_experiment.py`，300 例数据/真值/派生量及冻结源码核查通过，八份报告精确重算一致，四数值方法各六例重放一致，两版视觉共 1,200 条正式记录与图片哈希通过核查。摘要保存在本地 `runs/audit-v1/summary.json`，详见[全程审查](10-experiment-audit.md)。这些检查不等于独立性能验证。
- 主要限制：Pilot 已用于校准、选型和提示修改；SR 的精确命中为 Spike 302/302、Step 15/141，PELT 对 Step 为 91/141；Theil–Sen 的高 Affiliation 不等于边界重叠高。与新版视觉直接取并集仍有明显误报代价。下一步建议先完成输入隔离/冻结推理入口，再做固定候选复核及独立确认，均尚未实施。
- 纠正：P5 检测函数只接收 input，但运行器在预测前用 `verify_pilot()` 读取 truth 作完整性校验，原“评价前才读真值”表述已在 P5 协议修正；未发现标签进入检测计算。本次仅在 CLI 增加已有数值结果拒绝覆盖检查，不改冻结生成器、检测器、提示或评分；直接调用历史运行函数仍无覆盖保护。
- 验证：本轮 CLI 覆盖保护与文档/API 相关 9 项 pytest、Ruff、`git diff --check` 通过。审查原稿已加入 README 与网页文档目录。

## 2026-09-27 · 多版本评价与 P7a 提示语义实验完成

- 类型：用户要求、实现与运行。用户要求保留 Point 精确/±3 日及 Range Affiliation/逐日 IoU 等论文材料，并实际测试更明确的活动语义。新增 `comparison-views-v1`，保留四数值方法、旧视觉和新 `visual-semantics-v2` 的八行方法×任务结果；总体、分组、逐案例 JSON/CSV、成对成功诊断和 PNG/SVG 输出到本地 `runs/comparison-views-v1/`。检测方法版本与评价视图明确分离，P4 主评分及旧预测不改。
- 执行：本轮是原 PNG/模型/JSON Object/参数下仅改提示的全 300 例实验，取代此前先 schema A/B×60 例的草案顺序。12 次 Pilot 外预检通过，600 次正式调用已完成。Point 精确 F1 0.0837、±3 日 F1 0.4770、FAR 0.3077；Range Aff F1 0.8730、正轴平均日 IoU 0.5400、FAR 0.2446。结果改善以减少过报和长期区间边界为主，Point 精确定位仍弱于 SR，Normal Range 成功轴 FAR 仍达 0.5802；不宣称全面优于数值。
- 可靠性与成本：Point 297/300 成功（2 个数组结构失败、1 个超时），Range 281/300（19 个数组结构失败），全部失败保留且不重试。记录到 599 次 usage 共 1,441,389 tokens，一次超时费用未知，实际金额 null；正式调用计时 269.410 秒。配置、源码、预测及指标冻结于 `configs/p7a-frozen.json`，本地保留源码/config/依赖 ZIP 快照。296 项 pytest、Ruff 通过，新视图重现全部八份 P4 主报告。
- 限制与后续：这是受已观察 Pilot 启发的单次开发实验；两版调用时间不同，不足以排除服务漂移/随机性。schema 对照、候选复核、Agent 与独立确认集均未执行。完整表格、来源、分母和复现入口见 [多版本实验记录](09-versioned-comparison.md)。

## 2026-09-27 · P6 后诊断与 P7 研究设计

- 类型：离线分析、待实施设计。按用户“先论文调研再设计”的要求核查 VisualTimeAnomaly v2/TSAD-Agents、VLM4TS、AnomLLM、TAMA、AnomSeer、JSONSchemaBench 及 Qwen 官方结构化输出说明。`scripts/analyze_p6.py` 校验冻结输出并重现已有五份报告；本次没有模型 API 调用、数据重生成或历史结果修复。详细证据与来源统一见 [P7 设计](08-p7-design.md)。
- 诊断：Visual Point 的精确/±3 日 Recall 为 0.1061/0.6230；与 SR 并集只增加 10 TP，却新增 1322 FP。Single Step 的 23 个合法 Visual Range 输出全部在事件轴报警，21 个延伸到末日，支持优先检查活动过程与残留偏移的任务语义。Theil–Sen 的 Affiliation F1 0.9345 不等于边界准确，其正轴逐日 IoU 为 0.3398；Visual 为 0.3719，Single Transient 上视觉有更好的 IoU。以上容差/IoU/并集仅为事后诊断，不替换 P4 分数。
- 当时方案（P7a 顺序已被上方执行记录替代）：先做 schema-only 与 schema+任务定义的纯视觉对照，再做 SR/Theil–Sen 与新版独立视觉候选的固定 ID 复核。计划先在元数据分层的 60 例开发子集检查，再冻结方法并建立独立确认集。候选复核和确认数据仍未实施；最新前置条件见全程审查，不得把 Pilot 剩余案例称为未见测试。

## 2026-09-27 · P6 纯视觉基线运行与冻结

- 类型：实现、验收。`gnss-sim visual` 增加 `preflight`、`run`、`evaluate` 三阶段；`renderer-v1` 只从 `CaseInput.displacement_mm` 生成固定 N/E/U PNG，`visual-v1` 对 Point/Range 各调用一次 `qwen3.8-flash` 并严格验证 N/E/U JSON。推理阶段只校验 manifest/summary/输入哈希，不打开 truth；单独评价阶段复用 P4。6 条人工工程序列的 12 次请求全部合法，正式 Pilot 600 次请求产生两份各 300 行预测；图像及返回模型标识通过预检。冻结配置、源码/产物哈希见 [P6 冻结记录](../configs/p6-frozen.json)，原始响应、PNG、预测和报告保存在不入 Git 的 `runs/p6/qwen3.8-flash/`。
- 开发结果：Point P/R/F1/FAR/成功率为 0.0339/0.1061/0.0514/0.6213/0.96；Range Affiliation P/R/F1/FAR/成功率为 0.7227/0.8204/0.7573/0.5506/0.91。39 条 HTTP 200 回复是 JSON 数组而非冻结的 N/E/U 对象，Point 12 条、Range 27 条均按失败计入，不作修复或补跑。P5 的 Point SR 和 Range Rolling Theil–Sen 在同一开发 Pilot 分别为 F1 0.5518 与 Affiliation F1 0.9345；本结果不支持 P6 优于数值基线，也不代表独立测试。完整解释见 [P6 协议](07-p6-visual-baseline.md)。
- 运行成本与限制：正式请求总耗时 227.001 秒，输入/输出共 1,337,785 tokens，API 未返回实际费用，金额记 `null`。Pilot manifest/summary SHA256 未变，输入逐例哈希复核通过；模型服务未暴露稳定权重修订，响应只确认 `qwen3.8-flash`。P6 到此停止，后续只能使用冻结产物做 P7 设计或离线错误分析，不按开发分数更改 P6。

## 2026-09-27 · P6 主视觉模型改为 qwen3.8-flash

- 类型：用户修正、设计决定。P6 主模型固定使用现有 GNSS 项目的 `qwen3.8-flash`，显式 `enable_thinking=false`；2026-09-26 设计中建议的 Qwen2.5-VL-7B-Instruct 已被本条替代，不进入 P6。请求 ID 不受旧 `QWEN_MODEL` 环境变量覆盖。模型说明、端点预检和运行记录要求见 [P6 协议](07-p6-visual-baseline.md)。
- 当时状态：此条仅更新实施前设计，尚未调用模型或生成视觉预测；后续实施和结果见上方 P6 运行条目。当时约定正式 Pilot 前核实本机端点的图像输入、返回模型 ID、缩放及请求参数，若不可用则停止而不按 Pilot 结果选替代模型。

## 2026-09-26 · P6 纯视觉基线实施前设计

- 类型：设计决定、文档同步。P6 固定为 `pilot-v1` 上独立的 Point/Range 零样本图像基线：从 `CaseInput.displacement_mm` 渲染 N/E/U 三联图，每任务每例一次视觉模型请求，严格解析为现有 P4 结果模型，仍用原评价器计分。旧文档中的 NEUH 四联图已收束为 NEU；不得加入 H、数值摘要、P5 候选、few-shot 或 Agent。正式设计见 [P6 协议](07-p6-visual-baseline.md)。
- 真值隔离：P6 推理阶段只核对 manifest 指定的输入哈希，不调用会读取 `truth.json` 的 `verify_pilot()`；两任务各生成 300 行结果后单独评价。超时、API/解析失败保留失败行，原始模型响应另存且不入 Git。先在 Pilot 外做图像和 JSON 工程预检，随后冻结图像、提示、模型与运行配置。
- 当时状态与待决：仅完成协议和网页文档导航同步，没有 renderer、runner、模型端点核验、正式调用或 P6 分数。当时建议的 Qwen2.5-VL-7B-Instruct 已由上方 2026-09-27 决定替换；模型端点、响应标识、图像缩放和可用请求参数仍须实施前核实。P5 开发结果保持不变，不据此推断 P6 或独立测试表现。

## 2026-09-26 · P5 四个数值基线运行与方法冻结

- 类型：实现、验收。`gnss-sim numerical --method` 支持 SR、PELT、Matrix Profile、Rolling Theil–Sen；只读取固定 Pilot input，30 个 Normal 的逐轴最大分数校准三项阈值，PELT 从预定六个 beta 候选中按 Normal 误报选取，四项分别以现行 P4 evaluator 评分。300 例×四方法均执行成功；预测、report、run 记录在忽略入 Git 的 `runs/p5/`，四方法参数及选中方法另存[冻结配置](../configs/p5-frozen.json)。没有数值 ensemble、视觉模型或 Agent。
- 开发结果：Point SR 的 P/R/F1/FAR 为 0.4490/0.7156/0.5518/0.2101，PELT 为 0.4194/0.2054/0.2758/0.1458；Range MP 的 Affiliation P/R/F1/FAR 为 0.0939/0.0954/0.0939/0.0499，Rolling Theil–Sen 为 0.9284/0.9449/0.9345/0.1954。按预定 F1 优先规则，冻结 Point SR 和 Range Rolling Theil–Sen。PELT 三轴均回退最大 `β=32`，N/E/U 的 Normal 校准误报数为 3/2/0，没有为追求结果扩大候选集。完整口径与时间见 [P5 协议](06-p5-numerical-baselines.md)。同一 Normal 既用于校准又用于开发 FAR，所有结果仅说明开发 Pilot，不能推断独立泛化或现场预警。
- 验收：269 项 pytest、Ruff、`uv lock --check` 通过；四份 JSONL 各 300 条且零解析错误，SR 重跑的预测 SHA256 相同。`pilot-v1` manifest/summary SHA256 仍为 `a2e43db3493f86b36d1b962126f70f462b2ee3f4bf711bdbd84b078d43c10e33` / `a88b4ca674fc3e122f48ba798d7898af2016e02ad4e24e6328f405c62a369007`，逐例输入与真值通过冻结校验。P5 在此结束；P6 在当时尚未开始，后续结果见本文件顶部。

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
