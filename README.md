# GNSS 模拟实验台

独立的纯模拟日尺度 N/E/U 实验线。P1～P3 生成器、P4 固定 Pilot 与 Point/Range 评价器、P5 四个独立数值基线均已实现。没有真实 GNSS 数据、视觉检测或 Agent。

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

网页顶部的「实验文档」提供章节目录、正文排版和数学公式渲染。后端的 `docs/` 是文档原稿，网页通过 `GET /api/docs` 和 `GET /api/docs/{slug}` 读取同一份 Markdown。阅读顺序：

1. [研究问题与实验路线](docs/01-research-design.md)：名词、研究边界和 P1～P9 的状态。
2. [P1 正常序列模型](docs/02-p1-normal-model.md)：annual/semiannual 背景、白噪声、坐标及 H/R3D 公式。
3. [数据契约与可复现流程](docs/03-data-and-reproducibility.md)：输入与真值隔离、seed、文件和 API。
4. [P2 事件与 P3 场景协议](docs/04-planned-methods.md)：五种事件公式与六种场景。
5. [P4 固定 Pilot 与 Point/Range 评价协议](docs/05-p4-pilot-evaluator.md)：300 例配额、预测契约和两项任务的评分规则。
6. [P5 数值基线与参数冻结](docs/06-p5-numerical-baselines.md)：四个方法、Normal 校准、开发结果和冻结选择。

CLI 可直接生成：

```powershell
uv run gnss-sim generate --seed 20260924 --count 54 --case-type all_scenarios
uv run gnss-sim pilot --seed 20260925
```

`data/generated/` 保存数据集 manifest、`cases/<case_id>/input.json` 与单独的 `truth.json`，默认不入 Git。输入只有日期、固定参考坐标、三轴观测及其确定性派生量；背景、噪声、相位、事件与注入贡献仅在真值文件中。相同主 seed、案例序号、类型与生成器版本产生相同案例内容，批次 ID 和创建时间不要求相同。旧版本批次不再由新接口读取。

`data/pilots/pilot-v1/` 是独立固定的 300 例开发集；再次运行 `pilot` 只校验，不重抽。未来方法统一读取这里的 `input.json`，分别输出 Point/Range JSONL：

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

完成四项后自动生成两张开发对照表和选择记录；后续使用的 Point SR、Range Rolling Theil–Sen 的参数见[冻结配置](configs/p5-frozen.json)。Pilot 是开发集，表中的 F1/FAR 不代表独立测试表现。

## 已实现口径

- `generator_version = "event-v6"`；固定 2025 年的 365 个连续日观测，参考坐标为 `(0,0,0) mm`。该版本包含 P3 场景与逐事件贡献；背景及单事件公式沿用 P2。
- Annual 幅值 N/E/U 为 `1.0/1.0/1.5 mm`，semiannual 为 `0.25/0.25/0.5 mm`，周期分母 `365.25` 日；各轴相位由案例 seed 独立派生。白噪声标准差为 `0.5/0.5/1.0 mm`。周期结构参考 GNSS 时间序列文献，这组数值是本实验为异常可辨识性采用的受控 benchmark 设定，不代表现场精度，也不宣称全部来自参考论文。
- `observed = P0 + normal_background + measurement_noise + injected_deformation + observation_artifact`；Normal 例的两项注入为零。无 AR(1)、flicker noise 或 secular deformation。H 与 R3D 分别是观测坐标相对 P0 的水平和三维偏移模长，并非累计路程。
- `case_type` 可选 normal、五类 P2 事件、六类 P3 场景、`all` 或 `all_scenarios`。P3 每例 2～6 个事件，长期形变最多一个，Spike 可重复，S6 至少跨两轴；`all_scenarios` 均衡分配六场景。Spike、Slow Trend 终值和 Acceleration 终值的绝对幅值为 N/E 4.5 mm、U 9.0 mm；Step 和 Transient Shift 为 N/E 3.75 mm、U 7.5 mm，不随噪声标准差变化。同一 case seed 的不同类型共享完全相同的背景/噪声，事件使用独立 seed。
- 网页按类型折叠案例，并默认从独立真值接口读取和显示事件标注；事件时间线可筛选、选择和查看独立贡献。该视图只供研究人员核查，不作为未来视觉方法的输入。
- `POST /api/datasets` 只接受 `seed`、`count`、`case_type`，不兼容旧两字段请求或旧数据格式。

运行 `uv run pytest` 和 `uv run ruff check .` 验证。关键决定记录在[项目状态](docs/project-status.md)。P5 已验收，下一阶段为独立视觉方法。
