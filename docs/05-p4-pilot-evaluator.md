# P4 固定 Pilot 与 Point/Range 评价协议

> 版本：2026-09-26 · `pilot-v1` 是开发用合成数据集，生成器仍为 `event-v6`。本次只替换预测契约与评价方式；没有运行检测算法、视觉模型或 Agent。

## 数据集冻结

在模拟仓库根目录执行 `uv run gnss-sim pilot --seed 20260925`。首次写入 `data/pilots/pilot-v1/{manifest.json,summary.json,cases/}`；再次执行只校验文件哈希、生成器源码哈希与真值，不覆盖数据。30 个 Normal、120 个 Single、150 个 Multi 共 300 例。五种 Single 各 24，每种在 N/E/U 各 8、每个 type×axis 正负各 4；六个 Multi 场景各 25。分层只使用生成前由 seed 决定的 type/axis/sign/scenario，不按观测曲线或检测结果筛选。这是开发覆盖率设计，不代表自然发生率或独立最终测试。

`manifest.json` 保存每例类型、seed、分层属性及 `input.json`/`truth.json` SHA256；`summary.json` 汇总配额。`CaseTruth.events[]` 仍是唯一 canonical 真值，不另存标签。`input.json` 是未来检测器可读的案例文件；真值、场景及分层信息只供实验管理和评价。P1～P3 生成公式、噪声、幅值、事件时长与六场景均未改。

冻结文件校验值：`manifest.json` SHA256 `a2e43db3493f86b36d1b962126f70f462b2ee3f4bf711bdbd84b078d43c10e33`；`summary.json` SHA256 `a88b4ca674fc3e122f48ba798d7898af2016e02ad4e24e6328f405c62a369007`。数据按仓库约定不入 Git；跨工作树复现应核对这些值及 manifest 中每例文件哈希。生成与真值校验仍逐例检查事件 profile、贡献聚合和观测组合等式。

## 两项任务与预测格式

全部索引从 0 开始，共 365 日，只允许 N/E/U 三轴。Point 真值由 Spike 日期与 Step 起点组成；Range 真值由 Slow Trend、Acceleration、Transient Shift 的活动区间组成。Step 后稳定偏移及长期事件结束后的残余偏移不延长真值活动区间。异常类型只用于从 `events[]` 确定任务真值，不要求方法预测类型。

Point JSONL 每行一个 `PointResult`：

```json
{"case_id":"case_0001","method":"visual","status":"success","predictions":{"N":[81,152],"E":[],"U":[239]}}
```

Range JSONL 每行一个 `RangeResult`，区间两端均包含：

```json
{"case_id":"case_0001","method":"visual","status":"success","predictions":{"N":[[100,188]],"E":[[220,233]],"U":[]}}
```

三轴键均必填，不能增加其他轴或事件元数据。Point 日期须为整数且 `0≤t<365`；Range 须满足整数 `0≤start≤end<365`。重复日期、重叠区间和相邻区间在二值向量中取并集。`status="failed"` 的预测内容不参与评分；缺行、解析失败或 API 失败均视为该任务的执行失败，不兼容旧 `DetectionResult.events[]` JSONL。

## Point：精确日期微评分

每例每轴把 GT 与预测日期各转为 365 位二值向量，再拼接全部 300 例×三轴。逐日同轴精确匹配，`TP/FP/FN` 从二值向量计算；没有 ±1/±3 日容差。主指标为 `Precision=TP/(TP+FP)`、`Recall=TP/(TP+FN)`、`F1=2TP/(2TP+FP+FN)`；零分母按 0 处理。失败的正样本向量视为空预测，产生 FN。该二值化及普通 P/R/F1 路线参考 [VisualTimeAnomaly 官方代码](https://github.com/mllm-ts/VisualTimeAnomaly/blob/7158c4ff05bc6fb27bf5045a1e8129318ca4a47d/src/result_agg.py)。

## Range：Affiliation 的 case×axis 宏评分

每例每轴把 GT 与预测闭区间都写为 `vector[start:end+1]=1`，然后使用原始 `affiliation.generics.convert_vector_to_events` 和 `affiliation.metrics.pr_from_events(events_pred,events_gt,Trange=(0,365))`。转换后的事件为半开区间；例如闭区间 `[100,188]` 转为 `(100,189)`。这里统一 GT 与预测端点，是本项目对 [VisualTimeAnomaly](https://github.com/mllm-ts/VisualTimeAnomaly/blob/7158c4ff05bc6fb27bf5045a1e8129318ca4a47d/src/utils.py) 上游转换细节的明确适配。

只对**有 Range GT** 的 case×axis 计算 Affiliation Precision、Recall 及其调和均值 F1；一个轴中可包含多个合并后的异常区间。成功但预测为空、或该案例执行失败的正轴记 `P=R=F1=0`。分别对所有正轴的 P、R、F1 取算术平均，得到三个宏指标；`affiliation_f1` 是逐轴 F1 的平均，而非宏 P/R 的调和均值。没有 GT 的轴不送入 Affiliation。

直接依赖 [Affiliation Metrics Python 实现](https://github.com/ahstat/affiliation-metrics-py/tree/8d8449858096bbade6a6e70848d05c9cc9b846fe)，`pyproject.toml` 与 `uv.lock` 固定 commit `8d8449858096bbade6a6e70848d05c9cc9b846fe`；本项目没有重写 Affiliation 数学公式。论文依据为 [Xu 等，*Can Multimodal LLMs Perform Time Series Anomaly Detection?*，arXiv 2502.17812v2](https://arxiv.org/abs/2502.17812v2)。按 case×axis 宏平均是本项目三轴评价的适配，不能表述为该论文原封不动的整套评测。

## 负轴、失败与命令

两项任务各自报告 `FAR`：在**成功执行且无该任务 GT** 的 case×axis 中，有至少一处预测的比例。30 个全 Normal case 的三轴都属于两项任务的负轴；其他案例中无该任务 GT 的轴也计入。执行失败的负轴不当作干净样本，而由 `execution_success_rate=成功案例数/300` 单独呈现；若没有成功负轴，FAR 为 `null`。所有失败的正轴仍以零预测计入主指标分母。这样缺失结果不会抬高召回或 Affiliation 得分。

```powershell
uv run gnss-sim evaluate --task point --predictions point.jsonl --method visual --out point-report.json
uv run gnss-sim evaluate --task range --predictions range.jsonl --method visual --out range-report.json
```

报告只含 `task`、对应三项主指标、`far`、`execution_success_rate`。无效 JSONL 行在 stderr 列出行号，其缺失案例按失败处理。重复 case ID、未知 case ID、method 不一致或任务结果模型不匹配会报错。旧一对一匹配、IoU 门槛、Onset/End MAE、CCR/MCR 和旧 report 格式均已移除，没有兼容层。手工样例覆盖精确日期、区间端点、上游 Affiliation 一致性、宏/微聚合、负轴 FAR、无预测与失败；工程验收不代表检测性能。后续的数值基线与开发结果见 [P5 协议](06-p5-numerical-baselines.md)。
