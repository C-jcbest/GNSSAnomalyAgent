# GNSS 模拟实验台

比较数值与视觉方法对合成日尺度 N/E/U 异常的识别能力。生成器复用 GutenTAG 和 TODS，支持 Normal、Global Extremum、Trend、Transient Mean Shift。

## 启动

需要 Python 3.11+、uv、Node.js 20+。在项目根目录运行：

```powershell
uv sync
uv run python scripts/setup_generator.py
npm --prefix web ci
npm --prefix web run build
uv run gnss-sim serve
```

打开 <http://127.0.0.1:18765>。生成器使用独立的 `.venv-generator/` 环境。

```powershell
uv run gnss-sim generate --seed 20260930 --count 24 --case-type all
uv run gnss-sim --help
```

## 文档

- [数据生成](docs/data-generation.md)：上游来源、异常参数和标签。
- [检测与评价](docs/detection.md)：校准、输入隔离、模型预算和评分。
- [开发与运行](docs/development.md)：目录职责、命令和检查。
- [项目状态](docs/project-status.md)：当前有效范围。
- [协作规范](AGENTS.md)。

全部旧实验结果已作废，当前没有有效检测成绩或预置校准参数。后续实验从新登记开始，不迁移旧结果。
