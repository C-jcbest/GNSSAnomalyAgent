# GNSS 模拟实验台

独立的纯模拟日尺度 N/E/U 实验线。当前只完成 P1：固定参考坐标的正常序列、生成记录和交互式浏览。没有真实 GNSS 数据、异常注入、检测运行或 Agent。

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
4. [后续异常、检测与评价方法](docs/04-planned-methods.md)：P2～P9 的待实施草案，不代表现有结果。

CLI 可直接生成：

```powershell
uv run gnss-sim generate --seed 20260923 --count 10
```

`data/generated/` 保存数据集 manifest、`cases/<case_id>/input.json` 与单独的 `truth.json`，默认不入 Git。输入只有日期、固定参考坐标、三轴观测及其确定性派生量；背景、噪声、相位和空事件列表仅在真值文件中。相同主 seed、案例序号与生成器版本产生相同案例内容，批次 ID 和创建时间不要求相同。旧 `sim-*` 180 日目录不再由新接口读取。

## P1 口径

- `generator_version = "normal-v1"`；固定 2025 年的 365 个连续日观测，参考坐标为 `(0,0,0) mm`。
- Annual 幅值 N/E/U 为 `2/2/3 mm`，semiannual 为 `1/1/2 mm`，周期分母 `365.25` 日；相位由案例 seed 派生。白噪声标准差为 `1.5/1.5/3.0 mm`。这些数值参考 [Khazraei 与 Amiri-Simkooei (2020) 的合成实验参数表](https://academic.oup.com/gji/article/224/1/257/5911580)，不宣称代表现场精度。
- `observed = P0 + normal_background + measurement_noise`；无 AR(1)、flicker noise 或 secular deformation。H 与 R3D 分别是观测坐标相对 P0 的水平和三维偏移模长，并非累计路程。
- P1 的 `events=[]`，不提前保存异常运动状态。网页的正常背景与噪声视图只用于检查模拟器，不作为未来检测输入。
- `POST /api/datasets` 只接受 `seed` 和 `count`，不兼容旧 preset、config 或 180 日格式。

运行 `uv run pytest` 和 `uv run ruff check .` 验证。关键决定记录在[项目状态](docs/project-status.md)。后续 P2～P9 按阶段另行实施，每阶段验收后暂停。
