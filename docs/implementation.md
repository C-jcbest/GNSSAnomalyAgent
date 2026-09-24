# 运行手册与结果工作台

更新：2026-09-23。当前仅使用固定北京时间每日15点、180天日网格。Python 3.11+；在仓库根目录运行`uv sync`，本机模型连接变量见`.env.example`，不得输出或提交凭据。所有数据、模型响应、标签和运行结果都在本地；无命令自动采集平台数据。

## 构建和比较

本地已构建`data/processed/daily-comparison-v1/`及仅原生缺测的`data/processed/daily-comparison-native-v1/`。如需从冻结的`daily15-v2`另建，输出必须选择**不存在的新路径**；原数据包只读。协议、种子、来源隔离和评价限制见[日同期比较协议](daily-comparison-protocol.md)。已完成的运行与数值结果见[项目状态](project-status.md)。

仅保留原生缺测的事后敏感性实验见[原生缺测协议](daily-native-only-protocol.md)。`--native-only`不新增缺测，亦不填补来源已有缺口；使用新目录及独立验证冻结，不得覆盖原144例运行：

```powershell
uv run gnss-exp build-daily-set --source data/processed/daily15-v2 --out data/processed/daily-comparison-native-v1 --native-only
uv run gnss-exp run --dataset data/processed/daily-comparison-native-v1 --config configs/daily-comparison-v1.json --split development --methods hampel cusum iforest visual daily_union --out runs/daily-native-dev-v1
uv run gnss-exp run --dataset data/processed/daily-comparison-native-v1 --config configs/daily-comparison-v1.json --split validation --methods hampel cusum iforest visual daily_union --out runs/daily-native-validation-v1
uv run gnss-exp freeze-daily --dataset data/processed/daily-comparison-native-v1 --validation-run runs/daily-native-validation-v1 --out artifacts/daily-native-v1/frozen-config.json
uv run gnss-exp run --dataset data/processed/daily-comparison-native-v1 --config artifacts/daily-native-v1/frozen-config.json --split test --methods hampel cusum iforest visual daily_union --out runs/daily-native-test-v1
```

```powershell
uv run gnss-exp build-daily-set --source data/processed/daily15-v2 --out data/processed/daily-comparison-new
uv run gnss-exp run --dataset data/processed/daily-comparison-v1 --config configs/daily-comparison-v1.json --split development --methods hampel cusum iforest visual daily_union --out runs/daily-comparison-dev-new
uv run gnss-exp run --dataset data/processed/daily-comparison-v1 --config configs/daily-comparison-v1.json --split validation --methods hampel cusum iforest visual daily_union --out runs/daily-comparison-validation-new
uv run gnss-exp freeze-daily --dataset data/processed/daily-comparison-v1 --validation-run runs/daily-comparison-validation-new --out artifacts/daily-comparison-new/frozen-config.json
uv run gnss-exp run --dataset data/processed/daily-comparison-v1 --config artifacts/daily-comparison-new/frozen-config.json --split test --methods hampel cusum iforest visual daily_union --out runs/daily-comparison-test-new
```

不得用`--limit`形成正式测试；配置改动要使用新协议、目录和验证轮次，不能覆盖旧运行。`--resume`只允许原运行身份未变化的未落盘记录继续执行，已有错误记录不会悄然重采样。视觉默认`qwen3.8-flash`、显式`enable_thinking=false`；模型API可以单独调用，不访问平台。当前冻结配置每例一次逻辑视觉调用，网络/结构纠错共用最多两次尝试。

## 查看与导出

```powershell
uv run gnss-exp serve
```

打开本地`http://127.0.0.1:8765/`；页面只读，无模型调用或平台回源。实验档案按开发/验证/测试筛选，默认只列每日15点结果，小时历史必须显式选择且不能混排。运行明细包含进度、方法、总体与注入类型/缺测分组指标、耗时和失败。日注入运行的**注入标注**页按来源窗、缺测状态列出尖峰、台阶、缓慢增长、加速增长、波动增大及未注入六种情景；三轴图用色带和加粗曲线标明受控目标，列出分量、索引、日期和已观测标注单元，可聚焦或看全窗并导出PNG/SVG。它不叠加方法预测，也不认证原生背景。案例证据页另显示标签/预测对照与组合轨迹；配置页展示冻结身份。数据快照展示未标注日历史、覆盖和缺口；指标标准页直接读取[指标文档](metrics-criteria.md)。支持PNG/SVG/PDF单图及CSV图包导出，导出的结果也不能脱离未知背景和相关来源限制解释。

新组合`daily_union`复用本例本轮已调用的视觉模型响应与Hampel结果，组合行的请求数及耗时只记录**额外执行增量**，不是端到端成本；完整组合成本需把两路组成分支合计。无新增注入情景下的报警率不是现场误报率；正常真值组F1为N/A；首尝试完成率、技术完成率与检测质量是不同问题。缺测位置不参与定位分数，未覆盖阳性仍计FN。

## 自检与历史

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src/gnss_anomaly/daily_dataset.py src/gnss_anomaly/experiments.py tests/test_daily_dataset_comparison.py
```

旧小时构建、Agent工程命令及历史日预实验协议不属于当前主排名；采集/小时收束仅是固定15点数据的上游。历史冻结运行与原源码均不被新实验覆盖；清理前资料和源码哈希见本地`research/local/cleanup-20260923-daily-restart/`及`research/local/daily-comparison-source-20260923/`。数据快照整理详见[数据准备](data-preparation.md)，完整命令选项用`gnss-exp <命令> --help`查看。
