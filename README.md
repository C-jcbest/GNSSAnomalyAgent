# GNSS 模拟实验台

独立的纯模拟日尺度 N/E/U 实验线。P1 固定正常背景与 P2 五种单轴单事件注入已实现，可浏览曲线、事件时间线和独立贡献。没有真实 GNSS 数据、检测运行或 Agent。

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
4. [P2 单事件生成与后续检测草案](docs/04-planned-methods.md)：P2 的正式公式与 P3～P9 的待实施部分。

CLI 可直接生成：

```powershell
uv run gnss-sim generate --seed 20260923 --count 10 --case-type slow_trend
```

`data/generated/` 保存数据集 manifest、`cases/<case_id>/input.json` 与单独的 `truth.json`，默认不入 Git。输入只有日期、固定参考坐标、三轴观测及其确定性派生量；背景、噪声、相位、事件与注入贡献仅在真值文件中。相同主 seed、案例序号、类型与生成器版本产生相同案例内容，批次 ID 和创建时间不要求相同。旧版本批次不再由新接口读取。

## 已实现口径

- `generator_version = "event-v4"`；固定 2025 年的 365 个连续日观测，参考坐标为 `(0,0,0) mm`。
- Annual 幅值 N/E/U 为 `1.0/1.0/1.5 mm`，semiannual 为 `0.25/0.25/0.5 mm`，周期分母 `365.25` 日；各轴相位由案例 seed 独立派生。白噪声标准差为 `0.5/0.5/1.0 mm`。周期结构参考 GNSS 时间序列文献，这组数值是本实验为异常可辨识性采用的受控 benchmark 设定，不代表现场精度，也不宣称全部来自参考论文。
- `observed = P0 + normal_background + measurement_noise`；无 AR(1)、flicker noise 或 secular deformation。H 与 R3D 分别是观测坐标相对 P0 的水平和三维偏移模长，并非累计路程。
- `case_type` 从 normal 与五类事件中选择；每例最多一个单轴事件。Spike、Slow Trend 终值和 Acceleration 终值的绝对幅值为 N/E 4.5 mm、U 9.0 mm；Step 和 Transient Shift 为 N/E 3.75 mm、U 7.5 mm，不随噪声标准差变化。同一 case seed 的不同类型共享完全相同的背景/噪声，事件使用独立 seed。网页点击「显示真值」后才读取事件和生成成分。
- `POST /api/datasets` 只接受 `seed`、`count`、`case_type`，不兼容旧两字段请求或旧数据格式。

运行 `uv run pytest` 和 `uv run ruff check .` 验证。关键决定记录在[项目状态](docs/project-status.md)。P3～P9 按阶段另行实施，每阶段验收后暂停。
