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

CLI 可直接生成：

```powershell
uv run gnss-sim generate --seed 20260923 --count 10
```

`data/generated/` 保存数据集 manifest、`cases/<case_id>/input.json` 与单独的 `truth.json`，默认不入 Git。输入只有日期、固定参考坐标、三轴观测及其确定性派生量；背景、噪声和事件真值仅在真值文件中。相同 seed、配置与案例序号生成相同案例内容，批次 ID 和创建时间不要求相同。

## P1 口径

- 180 个连续日观测，默认从 `2025-01-01` 起；参考坐标固定为 `(0,0,0) mm`。
- 默认白噪声标准差 N/E/U 为 `1.5/1.5/3.0 mm`，AR(1) 系数 `0.7`，创新标准差 `0.75/0.75/1.5 mm`，365 日周期幅值 `2/2/4 mm`。这些只是实验默认值，未经真实稳定站校准。
- `true_coordinate = P0 + background`，`observed = true_coordinate + white_noise + ar_noise`。H、D 分别是观测坐标相对 P0 的水平和三维偏移模长，并非累计路程。
- P1 所有事件列表为空；`active_motion` 和 `persistent_offset` 仅描述异常注入造成的运动或残余偏移，因此均为 false，不把正常周期背景计为异常运动。网页中的“无噪声轨迹”和生成成分只用于检查模拟器，不作为检测输入。

运行 `uv run pytest` 和 `uv run ruff check .` 验证。后续 P2～P9 按阶段另行实施，每阶段验收后暂停。
