# 纯模拟 GNSS 实验线约定

- 优先中文回答和编写文档。本分支只使用合成的日尺度 N/E/U 数据，不读取旧项目的平台快照、标签或运行产物。
- 分阶段实施：P1 正常序列，P2 单异常，P3 多异常，P4 固定 Pilot 与 Point/Range 评价器；每阶段验收后暂停。先比较数值与视觉方法，再依据证据考虑 Agent，不预设方法优劣。
- 始终将检测输入与模拟真值分开保存。检测主输出分别报告异常日期或活动区间及分量，不要求异常类型，不凭曲线声称仪器故障、真实坡体形变或滑坡成因。
- `event-v6` 固定背景：2025 年 365 日、P0=(0,0,0) mm、annual/semiannual 幅值 1.0/1.0/1.5 与 0.25/0.25/0.5 mm、365.25 日周期、白噪声标准差 0.5/0.5/1.0 mm。五类事件幅值固定为绝对 mm，不能由当前噪声 σ 动态换算；不得再将 `sigma_multiplier` 作为核心 truth 字段。数值是受控 benchmark 设定，不代表现场精度；不加 AR(1)、secular deformation 或兼容层。
- P2 单事件入口保留供配对对照；P3 只能按六种场景模板组合，单例 2～6 事件、长期形变最多一个，不叠加 Slow Trend 与 Acceleration。每事件贡献独立保存，聚合逐元素求和；场景与真值不进入检测输入。`case_type=all` 是 P2 检查比例，`all_scenarios` 是均衡 P3 验收集合，均非 P4 Pilot 或自然频率。研究人员网页默认读取独立真值并显示异常标注，未来检测器仍只能读取 input；P3 阶段未实现检测器、评价器、缺测或 Agent。正式定义见 `docs/04-planned-methods.md`。
- P4 已冻结 `pilot-v1`（seed 20260925，30 Normal/120 Single/150 Multi）；后续方法统一使用该批输入，不按曲线或检测表现筛选、重抽。`events[]` 是唯一真值。Point 评 Spike/Step 起点的精确日二值微 P/R/F1；Range 评其余三类活动区间的 Affiliation case×axis 宏 P/R/F1；各自报告成功负轴 FAR 和案例执行成功率。旧 event matching/IoU/CCR/MCR 契约已破坏性移除，不兼容旧预测或报告。规则以 `docs/05-p4-pilot-evaluator.md` 为准。
- P5 已实现并验收：SR/PELT 分别作为 Point 基线，Matrix Profile/Rolling Theil–Sen 分别作为 Range 基线；只用 Pilot 的 30 个 Normal 输入校准阈值/penalty，再用固定 P4 evaluator 评价四个独立方法。开发集按预定规则选择 Point SR、Range Rolling Theil–Sen，四方法参数见 `configs/p5-frozen.json`。不得以异常真值调阈值、添加数值 ensemble 或把开发集 FAR 称为独立误报保证。实现与结果见 `docs/06-p5-numerical-baselines.md`。
- P6 已实现并在固定 Pilot 运行：N/E/U 三联图、独立零样本 Point/Range 提示，模型为 `qwen3.8-flash` 且显式 `enable_thinking=false`；旧 `QWEN_MODEL` 不覆盖。每任务各 300 行结果，Point F1 0.0514/FAR 0.6213/成功率 0.96，Range Affiliation F1 0.7573/FAR 0.5506/成功率 0.91；39 条结构失败按协议计入。配置与产物哈希见 `configs/p6-frozen.json`，解释见 `docs/07-p6-visual-baseline.md`。推理阶段不读取 `truth.json` 或调用 `verify_pilot()`；不得根据开发结果重跑、换模型、改画法或补救无效 JSON。
- 全部位移使用毫米；H 与 R3D 是相对于固定初始坐标的偏移模长，不是路径长度。R3D 不与注入形变分量 D 混用。模拟参数只是实验设定，不能宣称代表现场 GNSS 精度。
- 生成的数据、真值、凭据、模型原始响应和运行产物默认不入 Git。网页仅访问本地实验目录，不自动连接平台或模型 API。
- 用户已要求保留多版本和补充评价视图。`comparison-views-v1` 并列报告 Point 精确/±3 日和 Range Affiliation/正轴逐日 IoU，±1/7 日及并集仍作诊断；这些是同一预测的不同评价口径，不是新 detector，不替换 P4。历史四数值、`visual-v1` 和 `visual-semantics-v2` 都保留，见 `docs/09-versioned-comparison.md`。
- P7a 已按用户指令完成同一 300 例的 prompt-only 语义实验，仍为 qwen3.8-flash、thinking=false、原 PNG/JSON Object/传输参数；600 次正式调用及 22 次失败均保留，无修复重试。配置与结果见 `configs/p7a-frozen.json`，不得按分数覆盖该版本。此前 `docs/08-p7-design.md` 的 schema A/B 未实施；固定候选复核已按后续 P7b 独立协议完成。整个 Pilot 已用于开发分析，不能把其任何子集称为未见测试。
- `docs/` 是实验说明原稿，网页从后端文档接口读取。实现新阶段或修改公式、字段、评价口径时，同步更新相应文档及阶段状态；草案与已实现内容明确区分。
- 全程审查与后续建议见 `docs/10-experiment-audit.md`。P5 的 `verify_pilot()` 在推理前读取 truth 作完整性校验，不能称进程级输入隔离；检测函数未使用真值。独立确认需新增输入专用运行器，复用已冻结阈值，不重新校准确认集 Normal。CLI 拒绝覆盖已有数值结果，禁止绕过保护直接覆盖历史目录。
- `docs/11-experimental-handbook.md` 是网页默认的论文式方法手册；P7b 第 8～10 节已同步为实施规格；60 例开发结果见 `docs/12-p7b-candidate-review.md`。原评分和历史协议优先；新 U/C 采用严格依赖成功，与此前失败回退数值的并集诊断分开。更新方法、指标或状态时同步手册相关节，避免阅读入口与具体协议矛盾。

- P7b `candidate-review-v1` 已完成：60 例固定元数据分层，复用归档 N/V，4 次工程预检、107 次正式复核均结构合法；Range 5 个上游失败留在分母。C 相对 U 的 Point 精确 F1 和 Range Affiliation F1 均下降，继续 Gate 未通过，不自动执行独立 300 例确认或扩展 Agent。配置/源码/产物哈希见 `configs/p7b-registered.json` 和 `configs/p7b-frozen.json`。只读输入入口为 `input_only.py` / `candidate_review.py`；新输出拒绝覆盖，不按结果重试或修补。
- P7b 离线误删检查见 `docs/13-p7b-error-analysis.md`：分组为事后关联分析，禁止将来源/类型/阶段真值转成运行时路由；Range 损失按唯一日集合计数，不能累加交叠候选。`docs/14-independent-confirmation-protocol.md` 仅为独立 N/V 确认草案，尚未生成或运行；P7b 停止条件不变。正式确认前必须补齐登记强制校验、逐例数值失败行、纯输入视觉入口、中断账本及配对 bootstrap，不能直接重用固定 pilot-v1 的运行器。

- 用户于 2026-09-28 指定后续模型最大回复 tokens=8192。新纯视觉调用使用 `configs/p8-visual-8k.json` / `visual_8k.py`（visual-semantics-v2-8k），保持 JSON Object 与本地 schema 校验；不修改历史 512/2048 参数或重新标记旧成绩。确认入口尚待实现，必须接新适配器；不得静默回退 token 预算或将预算变更称作旧 v2 原样确认。

- P8a `p8a-range-context-v1` 已完成 60 例同期 V0/V1/V2 Range 对照：6 次预检、172 次正式调用，统一 8192，原图和局部图复核共享 V0、允许修订边界。V2 IoU 0.5805 高于 V0 0.5088，但主 Affiliation F1/recall/成功率下降，未通过预登记 Gate，不自动扩大到 Pilot300、Agent 或独立确认。具体结果与局限见 `docs/15-p8a-visual-range-context.md`；请求与原图/局部图、本地响应、失败及成本都冻结。6 条结构失败均正常 stop，不能归因于回复预算不足。推理源码/配置按 `configs/p8a-registered.json` 固定，后续修改需新版本；不能覆盖历史或重发账本中未知尝试。

- 2026-09-28 用户随后明确选择完整 pilot-v1 300 例继续测试，P8a 全集扩展已完成；不解释为原 Gate 通过。使用 `p8a_full.py` 和 `runs/p8a-full/`，复用原 60 例冻结响应及失败，只新增其余 240 例（660 次请求），三组 Range 设置保持不变。V0/V1/V2 全集主 F1 0.7788/0.7621/0.7652，IoU 0.4591/0.4546/0.5284；保留完整 300 与两阶段分表，缓存和新增调用不同期。仍为开发测试，不自动进入 Agent 或独立确认；冻结索引 `configs/p8a-full-frozen.json`。
