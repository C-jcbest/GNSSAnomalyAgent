# P4 固定 Pilot 与事件评价协议

> 版本：2026-09-25 · `pilot-v1` 是开发用合成数据集，生成器为冻结的 `event-v6`。P4 只固定数据、验证真值和评价器；没有运行检测方法或视觉模型。

## 数据集与复现

在模拟仓库根目录执行 `uv run gnss-sim pilot --seed 20260925`。首次写入 `data/pilots/pilot-v1/{manifest.json,summary.json,cases/}`；再次执行只校验文件哈希、生成器源码哈希及真值，不覆盖既有数据。目录被 Git 忽略。30 个 Normal、120 个 Single、150 个 Multi 固定为 300 例；五种 Single 各 24，分别在 N/E/U 各 8，每个 type×axis 的正负各 4；六个 Multi 场景各 25。该分布是开发覆盖率设计，不是自然发生率，也不是独立最终测试。

Pilot seed 使用与网页临时批次不同的 P4 命名空间。Normal 与 Multi 的 seed 按组、场景和序号确定。Single 仅用 `event-v6` 已有的 shape/sign 子 seed 预测轴与方向，并逐个搜索直至填满预先定义的 type×axis×sign 配额；不读取观测曲线、不按视觉显著性、噪声或检测成绩筛选。`manifest.json` 保存每例类别、seed、分层属性和输入/真值 SHA256；`summary.json` 汇总配额、每个 stratum、多事件数、同轴/跨轴和事件类型。真值只在各例 `truth.json` 的 `CaseTruth.events[]`；没有第二套人工标签。

本地冻结产物校验值：`manifest.json` SHA256 为 `a2e43db3493f86b36d1b962126f70f462b2ee3f4bf711bdbd84b078d43c10e33`，`summary.json` SHA256 为 `a88b4ca674fc3e122f48ba798d7898af2016e02ad4e24e6328f405c62a369007`。manifest 内含每例文件哈希与生成器相关源码哈希；跨工作树复现时应先核对这些值，不把生成数据提交 Git。

每例校验 typed truth、日期/索引、事件 profile、逐事件贡献之和与观测组合等式。`input.json` 是未来检测器唯一可读的案例文件；`truth.json`、manifest 分层字段和 summary 只供实验管理及评价读取。P4 未改 P1～P3 数学公式或幅值。

## 派生标签语义

`gnss_sim.labels.event_masks(truth)` 从 `events[]` 在内存中导出 365×3 的 `active` 和 `effect` 布尔矩阵，轴顺序 N/E/U；重叠事件取并集。`active` 包括事件起止日（闭区间）。Step 只有起变日为 active；Slow Trend/Acceleration 的 90 日活动区间为 active。持久事件从起点至年末为 `effect`，包括活动结束后的残余偏移；非持久事件的 effect 与 active 相同。当前事件评价只使用活动事件的时间语义，不把稳定后的残余偏移算作持续活动。

## 统一预测与事件匹配

预测 JSONL 每行一个 `DetectionResult`，含 `case_id`、`method`、`status=success|failed`、`events[]`。每项含 `prediction_id`、`axes`、`start_index`、`end_index`、可选 `confidence`；`type` 可选，评价器完全忽略它。索引从 0 开始、闭区间。点/突变起点须用 `start_index=end_index`；持续演化须用 `start_index<end_index`。预测严格单轴且轴与真值一致；三轴全报不匹配单轴真值。

候选边同时要求轴和时间兼容。Spike 为单日预测，起点误差 ≤1 日；Step 为单日预测，起点误差 ≤3 日。Slow Trend、Acceleration、Transient Shift 均要求区间预测且闭区间 tIoU ≥0.5，交集和并集日数均含端点。匹配严格一对一：先最大化 TP 数量，同 TP 数量时最大化归一化质量（点事件 `1−|起点误差|/(容差+1)`，区间事件为 tIoU）。排序与匹配不依赖输入事件顺序；ID 只用于追溯配对，不参与评分。

`uv run gnss-sim evaluate --predictions predictions.jsonl --method METHOD --out report.json` 读取固定 Pilot 并输出逐例与汇总结果。缺行视为执行失败，已标记 `failed` 的预测列表不计入事件分数；两者均保留案例分母。无法解析的 JSONL 行记入 `invalid_result_lines`，其缺失案例仍按失败计；重复 case ID、未知 case ID 或 method 不一致会报错。

## 汇总指标

- 全 300 例微汇总 `TP/FP/FN`，`Precision=TP/(TP+FP)`、`Recall=TP/(TP+FN)`、`F1=2TP/(2TP+FP+FN)`；分母为零时为 `null`，不伪造完美分数。
- 已匹配事件的 `OnsetMAE`（天）；已匹配区间事件的 `MeanIoU` 与 `EndMAE`（天）。无匹配时相应值为 `null`，须结合 Recall 解读。
- Normal FAR = 有至少一个有效预测的成功 Normal 案例数 / 成功 Normal 案例数；另报 FP/Normal 和 NormalFailureRate。失败 Normal 不计为 clean，成功 Normal 为零时 FAR 为 `null`。
- Multi Mean Case Recall = 每例 TP/真值事件数的均值；Multi Complete Case Rate = `FN=0` 且 `FP=0` 的案例比例。CCR 是本任务自定义完整识别指标，不称传统 TSAD 标准。
- ExecutionSuccessRate = `success` 案例数 / 300。失败的异常例按零预测计 FN，失败 Normal 单列失败率；所有失败都保留在 300 例中。

P4 的手工预测样例测试覆盖容差与 IoU 边界、严格轴、重复预测、一对一竞争、顺序/ID 不变性、Normal FAR、Multi CCR 和方法失败。工程验收不等于 Numerical、Visual 或组合的检测性能结果。P5 才开始方法开发。
