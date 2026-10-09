# GNSS 模拟实验台

比较数值与视觉方法对合成日尺度 N/E/U 异常的识别能力。生成器复用 GutenTAG 和 TODS，支持 Normal、Global Extremum、Trend、Transient Mean Shift。

另提供长期滑坡位移仿真：同一记录内组合静稳、缓慢形变、加速、减速、停滞与再活动，允许中途起变和首尾截断。支持时间相关误差、缺测、批量曲线和观测派生速度/加速度辅助图；这些数据是文献机制启发的仿真，不是现场实测。

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
uv run gnss-sim generate-landslides --seed 20261009 --count 24 --days 1095
uv run gnss-sim --help
```

## 文档

- [数据生成](docs/data-generation.md)：上游来源、异常参数和标签。
- [长期滑坡数据设计](docs/landslide-design.md)：调研依据、多年过程与展示口径；网页入口为“长期位移”。
- [检测与评价](docs/detection.md)：校准、输入隔离、模型预算和评分。
- [开发与运行](docs/development.md)：目录职责、命令和检查。
- [项目状态](docs/project-status.md)：当前有效范围。
- [协作规范](AGENTS.md)。

实验登记、LangChain 工具调用和对照命令见[开发与运行](docs/development.md)，当前结论与本地报告入口见[项目状态](docs/project-status.md)。
