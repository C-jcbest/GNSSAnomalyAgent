# GNSS 模拟实验台

独立的纯模拟日尺度 N/E/U 实验线。P1～P3 生成器、P4 固定 Pilot 与 Point/Range 评价器、P5 四个独立数值基线、P6 纯视觉基线及 P7a 提示语义对照均已实现。保留多版本预测和多评价视图，没有真实 GNSS 数据或 Agent。

## 启动

需要 Python 3.11+、uv、Node.js 20+。在本工作树根目录运行：

```powershell
uv sync
cd web
npm install --cache .npm-cache --registry=https://registry.npmjs.org/
npm run build
cd ..
uv run gnss-sim serve --port 18765
```

浏览器打开 `http://127.0.0.1:18765`。若端口已占用，`serve --port <可用端口>` 可使用其他端口。开发时也可在后端使用默认 `8765` 端口的前提下运行 `npm run dev`，前端地址为 `http://127.0.0.1:5173`。

拉取或切换到包含新接口的代码后，应停止旧的 `gnss-sim serve` 进程并重新启动；旧进程不会自动加载新接口，可能使「实验文档」显示 404。

网页的「实验文档」默认打开[实验方法手册](docs/11-experimental-handbook.md)：按论文结构解释任务、名词、公式、指标分母、既有结果及 P7b 四条件设计。可直接访问 `http://127.0.0.1:18765/?view=docs&doc=experimental-handbook`。支持文档全文搜索、两级页内目录与阅读进度、文内协议跳转、章节链接分享、原稿下载、打印及移动端折叠菜单。

后端的 `docs/` 是文档原稿，网页通过 `GET /api/docs`（slug/title/file）和 `GET /api/docs/{slug}` 读取同一份 Markdown。方法手册之后可按需查阅：

1. [研究问题与实验路线](docs/01-research-design.md)：名词、研究边界和 P1～P9 的状态。
2. [P1 正常序列模型](docs/02-p1-normal-model.md)：annual/semiannual 背景、白噪声、坐标及 H/R3D 公式。
3. [数据契约与可复现流程](docs/03-data-and-reproducibility.md)：输入与真值隔离、seed、文件和 API。
4. [P2 事件与 P3 场景协议](docs/04-planned-methods.md)：五种事件公式与六种场景。
5. [P4 固定 Pilot 与 Point/Range 评价协议](docs/05-p4-pilot-evaluator.md)：300 例配额、预测契约和两项任务的评分规则。
6. [P5 数值基线与参数冻结](docs/06-p5-numerical-baselines.md)：四个方法、Normal 校准、开发结果和冻结选择。
7. [P6 纯视觉基线与冻结结果](docs/07-p6-visual-baseline.md)：固定 `qwen3.8-flash`、N/E/U 三联图、独立 Point/Range 运行和开发结果。
8. [P6 后诊断与 P7 设计](docs/08-p7-design.md)：冻结结果错误分析、论文依据及候选复核草案；实际执行调整见下一章。
9. [多版本方法与提示语义实验](docs/09-versioned-comparison.md)：保留原预测，统一导出 Point 精确/±3 日与 Range Affiliation/逐日 IoU；新版视觉提示的执行协议与结果。
10. [实验全程审查与下一阶段建议](docs/10-experiment-audit.md)：可复现检查、方法覆盖与指标限制、P7b 最小复核实验及独立确认前置条件。
11. [P7b 固定候选复核](docs/12-p7b-candidate-review.md)：60 例 N/V/U/C 对照、严格失败、输入隔离及负结果；复现命令见文档第 8 节。
12. [P7b 错误归因与实验链检查](docs/13-p7b-error-analysis.md)：按来源、形态、轴、场景和活动阶段分解误删，核验冻结结果与确认工程缺口。
13. [独立合成确认协议（草案）](docs/14-independent-confirmation-protocol.md)：独立 N/V 的新 300 例比较、失败账本、配对 bootstrap 和执行 Gate；视觉新增 `visual-semantics-v2-8k` 配置（回复上限 8192，原提示），尚未生成或运行。
14. [P8a 视觉 Range 全局与局部对照](docs/15-p8a-visual-range-context.md)：60 例对照及用户指定的完整 Pilot300 测试均已完成；局部图提高日 IoU，但主 F1 未超过单轮，独立确认与 Agent 未启动。含 8192 tokens、缓存/新增调用、失败账本和两阶段分表。

CLI 可直接生成：

```powershell
uv run gnss-sim generate --seed 20260924 --count 54 --case-type all_scenarios
uv run gnss-sim pilot --seed 20260925
```

`data/generated/` 保存数据集 manifest、`cases/<case_id>/input.json` 与单独的 `truth.json`，默认不入 Git。输入只有日期、固定参考坐标、三轴观测及其确定性派生量；背景、噪声、相位、事件与注入贡献仅在真值文件中。相同主 seed、案例序号、类型与生成器版本产生相同案例内容，批次 ID 和创建时间不要求相同。旧版本批次不再由新接口读取。

`data/pilots/pilot-v1/` 是独立固定的 300 例开发集；再次运行 `pilot` 只校验，不重抽。检测方法统一读取这里的 `input.json`，分别输出 Point/Range JSONL：

```powershell
uv run gnss-sim evaluate --task point --predictions point.jsonl --method METHOD --out point-report.json
uv run gnss-sim evaluate --task range --predictions range.jsonl --method METHOD --out range-report.json
```

Point 为逐轴异常日期列表，按精确日期微 P/R/F1 评分；Range 为逐轴闭区间列表，按 Affiliation case×axis 宏 P/R/F1 评分。失败与缺失保留固定案例分母；完整字段与 FAR 口径见 [P4 协议](docs/05-p4-pilot-evaluator.md)。

P5 四个数值基线分别运行并保存到忽略入 Git 的 `runs/p5/`：

```powershell
uv run gnss-sim numerical --method sr
uv run gnss-sim numerical --method pelt
uv run gnss-sim numerical --method matrix-profile
uv run gnss-sim numerical --method theilsen
```

完成四项后自动生成两张开发对照表和选择记录；后续使用的 Point SR、Range Rolling Theil–Sen 的参数见[冻结配置](configs/p5-frozen.json)。CLI 拒绝覆盖已有预测、报告或运行记录；有计划的重现实验须指定单独的 `--out-dir`。Pilot 是开发集，表中的 F1/FAR 不代表独立测试表现。

P6 已按[冻结配置](configs/p6-visual.json)完成。首次执行时，先配置本地 `QWEN_BASE_URL`/`QWEN_API_KEY` 环境变量，或通过 `--env-file <本地 .env 路径>` 指定凭据文件，然后依次运行：

```powershell
uv run gnss-sim visual --phase preflight --env-file "C:\path\to\.env"
uv run gnss-sim visual --phase run --env-file "C:\path\to\.env"
uv run gnss-sim visual --phase evaluate
```

预检只使用 Pilot 以外的人工序列，正式阶段对同一 300 张 N/E/U 图分别执行 Point/Range 请求，评价另行读取真值。原始响应与预测留在忽略入 Git 的 `runs/p6/`；已有预检和正式运行不会被自动覆盖。当前结果与失败口径见[P6 协议](docs/07-p6-visual-baseline.md)。

## 已实现口径

- `generator_version = "event-v6"`；固定 2025 年的 365 个连续日观测，参考坐标为 `(0,0,0) mm`。该版本包含 P3 场景与逐事件贡献；背景及单事件公式沿用 P2。
- Annual 幅值 N/E/U 为 `1.0/1.0/1.5 mm`，semiannual 为 `0.25/0.25/0.5 mm`，周期分母 `365.25` 日；各轴相位由案例 seed 独立派生。白噪声标准差为 `0.5/0.5/1.0 mm`。周期结构参考 GNSS 时间序列文献，这组数值是本实验为异常可辨识性采用的受控 benchmark 设定，不代表现场精度，也不宣称全部来自参考论文。
- `observed = P0 + normal_background + measurement_noise + injected_deformation + observation_artifact`；Normal 例的两项注入为零。无 AR(1)、flicker noise 或 secular deformation。H 与 R3D 分别是观测坐标相对 P0 的水平和三维偏移模长，并非累计路程。
- `case_type` 可选 normal、五类 P2 事件、六类 P3 场景、`all` 或 `all_scenarios`。P3 每例 2～6 个事件，长期形变最多一个，Spike 可重复，S6 至少跨两轴；`all_scenarios` 均衡分配六场景。Spike、Slow Trend 终值和 Acceleration 终值的绝对幅值为 N/E 4.5 mm、U 9.0 mm；Step 和 Transient Shift 为 N/E 3.75 mm、U 7.5 mm，不随噪声标准差变化。同一 case seed 的不同类型共享完全相同的背景/噪声，事件使用独立 seed。
- 网页按类型折叠案例，并默认从独立真值接口读取和显示事件标注；事件时间线可筛选、选择和查看独立贡献。该视图只供研究人员核查，不作为 P6 模型输入。
- `POST /api/datasets` 只接受 `seed`、`count`、`case_type`，不兼容旧两字段请求或旧数据格式。

运行 `uv run pytest` 和 `uv run ruff check .` 验证。关键决定记录在[项目状态](docs/project-status.md)。P5/P6/P7a 已验收，P7b 开发运行已完成且未通过继续确认 Gate；`uv run python scripts/export_comparison.py` 从八份冻结方法×任务预测导出论文表格、分组/逐案例 CSV 和 PNG/SVG，输出到 `runs/comparison-views-v1/`。后续分析使用冻结的开发产物，不把 Pilot 结果当独立测试。
