# 开发与运行

## 目录职责

| 路径 | 职责 |
| --- | --- |
| src/gnss_sim/generator.py、schemas.py、storage.py | 生成编排、当前数据契约、批次存储 |
| src/gnss_sim/numerical.py、visual.py、rendering.py | 数值方法、视觉解析/凭据、无标签检测图 |
| src/gnss_sim/detection.py | 校准边界、运行登记、数值及视觉执行 |
| src/gnss_sim/metrics.py、report.py | 原生标签评分、离线报告 |
| src/gnss_sim/artifacts.py、api.py、cli.py | 文件读写、网页API、命令入口 |
| scripts/setup_generator.py、generate_worker.py | 独立生成环境安装与上游调用 |
| configs/、vendor/ | 上游版本/依赖/许可证与最小补丁 |
| web/src/ | 生成工作台、只读检测说明、文档阅读器 |
| tests/ | 生成、API、推理契约及评价的必要功能检查 |

## 运行命令

安装与网页启动见[README](../README.md)。网页端口默认18765；更新后端代码后重启服务。开发时运行 `uv run gnss-sim serve --port 8765`，另在 `web/` 运行 `npm run dev`。

下面是未来已登记实验的命令模板，不代表自动执行计划。先生成不同seed的Normal校准集和评价集；记录两次命令输出的实际数据目录，再填写路径：

```powershell
uv run gnss-sim generate --seed 1001 --count 100 --case-type normal
uv run gnss-sim generate --seed 2001 --count 24 --case-type all
uv run gnss-sim calibrate --dataset <Normal数据目录> --out <新参数文件.json>
uv run gnss-sim prepare --dataset <评价数据目录> --parameters <新参数文件.json> --out <新运行目录>
uv run gnss-sim numerical --run <运行目录>
uv run gnss-sim visual --run <运行目录> --env-file <本机凭据文件>
uv run gnss-sim evaluate --run <运行目录>
```

视觉命令产生外部调用；其余命令不调用模型。用浏览器打开运行目录中的 `comparison.html` 查看结果。网页不读取旧报告副本，也不会自动连接模型API。

## 验证

```powershell
uv run pytest -q
uv run ruff check .
npm --prefix web run build
```

生成测试需要先安装 `.venv-generator/`。推理测试使用MockTransport，不消耗模型额度；生成/校准/评分测试仅使用临时工程数据，不产生有效实验成绩。

仅为功能契约添加必要测试：真实上游组合、观测/真值隔离、失败分母与日期去重、请求上限与不重发、参数篡改与防覆盖。文案和样式修改不新增单元测试。

## 本地文件

`.venv/` 和 `.venv-generator/` 为可重建环境；`data/`、`runs/`、`artifacts/`、`papers/local/` 和凭据均不入Git。论文留在本地。不要在AGENTS.md堆积状态或历史记录；上游源码不要为满足lint而格式化。
