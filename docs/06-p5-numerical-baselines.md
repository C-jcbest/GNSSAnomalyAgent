# P5 数值基线与参数冻结

> 状态：2026-09-26 已实现并验收；P4 `pilot-v1`、`event-v6` 与 [Point/Range 评价协议](05-p4-pilot-evaluator.md) 保持冻结。本文的数值选择是本项目预先定义的实现约定，不是参考论文的原封不动复现。

## 目标与范围

P5 在同一批 300 例开发输入上运行四个可解释的离线基线：Point 使用 Spectral Residual（SR）和 PELT，Range 使用 Matrix Profile（MP）和 Rolling Theil–Sen。每个方法独立产出 P4 JSONL、评价报告和运行记录；完成后按预先确定的规则冻结一个 Point 方法和一个 Range 方法，供 P6/P7 比较使用。P5 不建立数值 ensemble，不调用视觉模型或 Agent，也不增加检测形态、调 P4 评分或重抽 Pilot。

[TSB-AD](https://papers.nips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html) 与其[公开代码](https://github.com/TheDatumOrg/TSB-AD)为 SR、MP 提供基线参照；本项目仅实现小型 SR 分数函数，MP 直接调用 STUMPY，不引入整套 benchmark。Point/Range 分任务的研究动机来自 [Xu 等的 VisualTimeAnomaly 论文 v2](https://arxiv.org/abs/2502.17812v2)。这些来源不保证所选方法在本数据上领先，P5 结果只能作开发集能力分析。

| 任务 | 方法名 | 输入与分数 | 输出 |
| --- | --- | --- | --- |
| Point | `sr` | 每轴 SR saliency，越大越异常 | 超阈值的日期索引 |
| Point | `pelt` | 每轴 L2 分段的内部断点 | 断点对应的日期索引 |
| Range | `matrix-profile` | 每轴 30 日子序列的最近邻距离，越大越异常 | 超阈值连续日期的闭区间 |
| Range | `theilsen` | 每轴 30 日窗口斜率绝对值，越大越异常 | 超阈值连续日期的闭区间 |

## 共同输入和时间约定

方法只接收 `CaseInput.displacement_mm` 的 N/E/U 三条**带符号** 365 日序列。`CaseTruth`、事件类型、场景、manifest 中的分层属性、注入成分及 H/R3D 都不进入检测函数。Normal 校准集合由固定 Pilot manifest 的 `case_type=normal` 标识，校准只读取这 30 例的 `input.json`；评价器在产生全部预测之后独立读取真值。所有方法按 N/E/U 分别运行，不做跨轴共享预测或基于 GT 类型的路由。

所有日期为 0 起始索引。Point 不移动峰值、不设日期容差或 NMS；Range 对 `score_t>τ_c` 的逐日布尔序列直接提取极大连续段，输出两端均包含的 `[start_index,end_index]`，不做最短长度、间隙合并或人工扩展。分数相等于阈值时不报警。无分数的边缘日期固定为 0，不以最近分数填充。输入异常、非有限分数或库运行失败按该案例该方法的 `status="failed"` 记录，不能删掉案例；实现时用 P4 schema 验证每行。

## 四个方法的固定定义

### SR → Point

对每轴先减全年的中位数，再计算 FFT 的振幅和相位。以 `ε=1e-12` 防止 `log(0)`，对 log 振幅按频率索引做宽度 3 的循环均值平滑：

$$
L_k=\log\max(|\mathcal F(x-\operatorname{med}(x))_k|,\epsilon),\qquad
R_k=L_k-\operatorname{MA}_3(L)_k,
$$
$$
score_t=\left|\mathcal F^{-1}\bigl(\exp(R_k+i\arg\mathcal F(x-\operatorname{med}(x))_k)\bigr)_t\right|.
$$

常数序列的分数定义为全零。该实现只取 [Microsoft SR 工作](https://arxiv.org/abs/1906.03821)中的频谱残差思路，不包含其 CNN、预测尾点、局部 saliency 再平滑或论文评分规则；也不直接复制 [TSB-AD 的 SR 实现](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/models/SR.py)。分数长度必须为 365，各轴以 Normal 最大值校准 `τ_SR,c`。

### PELT → Point

每轴先用全年中位数与 MAD 做固定鲁棒标准化：

$$
z_t=\frac{x_t-\operatorname{med}(x)}{\max(1.4826\operatorname{MAD}(x),10^{-9})}.
$$

调用 `ruptures.Pelt(model="l2", min_size=3, jump=1).fit(z).predict(pen=β_c\log 365)`。`β_c` 按下文仅在 Normal 上校准。`ruptures` 的断点 `b` 表示新片段 `[b,\ldots)` 的第一个索引，因此内部断点直接输出为 Point 日期 `b`；末尾的 `365` 是序列终点，必须删除。人工无噪声 Step 从索引 120 起时，应检验内部断点与输出均为 120，不能凭性能结果修正 ±1。[ruptures 文档](https://centre-borelli.github.io/ruptures-docs/examples/basic-usage/)说明末尾索引约定；算法依据见 [Killick 等的 PELT 论文](https://arxiv.org/abs/1101.1438)。

### Matrix Profile → Range

对每轴调用 `stumpy.stump(x, m=30, normalize=True)`，取第一列 z 标准化最近邻距离。365 日输入得到 `365-30+1=336` 个子序列分数。第 `i` 个窗口 `[i,i+29]` 固定映射至**右中心日** `t=i+15`；其余边缘日分数为 0，不把高分窗口涂满 30 天。这个日映射是本项目的评分适配，须和 Normal 校准、正式预测完全一致。[STUMPY 文档](https://stumpy.readthedocs.io/en/latest/api.html)明确 profile 长度、默认自连接和 z 标准化距离；[TSB-AD wrapper](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/models/MatrixProfile.py)仅作实现参照。窗口 30 日预先固定，不按 14/90 日 GT 时长增设多尺度组合。

### Rolling Theil–Sen → Range

同样对每轴的每个 30 日窗口 `[i,i+29]`，使用 `scipy.stats.theilslopes(y=x[i:i+30], x=np.arange(30)).slope`，分数为斜率绝对值，单位 mm/日；同样映射到 `t=i+15`，边缘为 0。它是透明的趋势基线，不声称专门发现所有 Range 类型。[SciPy 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html)说明斜率为成对斜率的中位数。范围提取与 MP 完全共用上述布尔序列规则。

## 只在 Normal 上校准

SR、MP、Theil–Sen 各自对 30 个 Normal 案例的每轴计算 `M_{i,c}=max_t score_{i,c,t}`。轴阈值固定为 `numpy.quantile(M[:,c], 0.95, method="linear")`；N/E/U 各存一个值，超过阈值才报警。不读取其他 270 例的真值或分数来改阈值，也不搜索 F1。由于同一批 Normal 同时参与校准及 P4 FAR 统计，这个 FAR 是**开发集内描述值**，不能解释为独立数据上的 5% 误报保证；30 个最大值的 95% 分位数也不保证观测 FAR 恰好为 5%。

PELT 对每轴分别从固定候选 `β∈{1,2,4,8,16,32}` 按升序选取首个使 30 个 Normal 中至多 1 例产生内部断点的值，即该轴经验误报率不超过 `1/30≈3.33%`。如果无候选满足，直接用 32 并记录实际误报，不扩展候选集。完整配置保存三个轴各自的 `β` 与实际 `pen=β log 365`。上述校准规则不能根据异常案例表现改变。

## 运行、选择与冻结

当前只增加 `gnss-sim numerical --method {sr,pelt,matrix-profile,theilsen}` 一个 CLI 子命令；方法与 P4 task 固定映射，不由用户另传 task。`numerical.py` 保留四个分数/断点函数及小型转换函数；`numerical_runner.py` 负责读取固定 Pilot、Normal 校准、全量预测、调用原 P4 evaluator 与保存文件。只增加 `scipy`、`ruptures`、`stumpy` 并用 `uv.lock` 固定实际版本，不引入 TSB-AD、Torch 或插件框架。

输出置于不入 Git 的 `runs/p5/<method>/`：`predictions.jsonl`、`report.json`、`run.json`。JSONL 按 Pilot manifest 案例顺序、N/E/U 轴顺序和升序索引稳定序列化；失败也写一行。`run.json` 记录 task、方法、完整参数与校准阈值、三轴 Normal 校准误报数、Pilot manifest/summary SHA256、源码修订与依赖版本、失败数和 `runtime_seconds`。计时从读取第一例输入前开始，包含 Normal 校准、300 例推理和预测写盘，不含 P4 评价与报告生成；STUMPY 首次 JIT 开销计入。时间只作相同环境下的描述量，不作选型平局条件。运行前后校验 Pilot 全部输入与真值哈希；不得写入 `data/pilots/pilot-v1/`。

两张开发表分别列 Point 的 P/R/F1/FAR/执行成功率/时间和 Range 的 Affiliation P/R/F1/FAR/执行成功率/时间。四个方法必须先达到 300/300 案例执行成功才进入选型。Point 按 P4 `f1`、Range 按 P4 `affiliation_f1` 从高到低选；平分时选 FAR 较低者，再平分按固定顺序 `sr` 优于 `pelt`、`matrix-profile` 优于 `theilsen`。选择结果与两个方法的完整配置写成冻结记录，后续 P6/P7 不按视觉结果或独立测试结果重选。Pilot 是开发集，所选胜者的 Pilot 分数不能作为独立泛化结论。

## 验收门槛

1. 纯函数相同输入产生相同 365 日分数；MP 与 Theil–Sen 的 336→365 日映射、边缘、Range 两端含义有人工样例。
2. PELT 人工 Step 索引 120 的断点转换为 120，terminal 365 不输出；Normal penalty 候选按固定顺序选择。
3. 校准只访问 30 个 Normal `input.json`，检测函数不接受 `CaseTruth`；阈值和 `β` 的三轴配置可从运行记录复建。
4. 300 例各有合法 `PointResult` 或 `RangeResult`，失败不删行；同配置重跑的 JSONL 字节一致，P4 evaluator 接受原样预测。
5. Pilot manifest、summary、全部 `input.json`/`truth.json` 在 P5 前后哈希一致；只输出两张开发表和两个冻结方法，不声明模型优劣或现场预警能力。

## 开发 Pilot 结果与冻结选择

四个方法均产生 300 条合法预测，执行成功率 1.0。Point 的主指标为 P4 精确日微评分；Range 的主指标为 P4 case×axis Affiliation 宏评分。FAR 按 P4 成功负轴定义，包含其他任务有 GT、而当前任务无 GT 的轴；30 个 Normal 又参与了阈值校准，因此这些数字都不是独立测试结论。时间为本地一次全量运行的秒数，含校准和预测写盘，不含 P4 评价。

| Point 方法 | Precision | Recall | F1 | FAR | 时间（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| SR | 0.4490 | 0.7156 | **0.5518** | 0.2101 | 0.27 |
| PELT | 0.4194 | 0.2054 | 0.2758 | 0.1458 | 587.46 |

| Range 方法 | Affiliation P | Affiliation R | Affiliation F1 | FAR | 时间（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matrix Profile | 0.0939 | 0.0954 | 0.0939 | 0.0499 | 21.74 |
| Rolling Theil–Sen | 0.9284 | 0.9449 | **0.9345** | 0.1954 | 62.70 |

PELT 六个候选 penalty 均未让 N/E 的 30 个 Normal 误报例数降到预设的至多 1 例，故按预注册回退使用三轴 `β=32`；此时 N/E/U 的 Normal 误报例数分别为 3/2/0，没有扩展搜索范围。三个分数法按 95% 分位数校准后，每轴 30 个 Normal 中均有 2 例超过阈值，符合已说明的有限样本分位数语义。

依预定 F1 优先规则，P5 冻结 **SR 为 Point 方法、Rolling Theil–Sen 为 Range 方法**。四个方法的完整固定参数、三轴校准值、依赖版本、Pilot 哈希和源码哈希保存在[受版本控制的冻结配置](../configs/p5-frozen.json)；逐例预测和原始报告保存在不入 Git 的 `runs/p5/`。Pilot manifest/summary 哈希仍分别为 `a2e43db3493f86b36d1b962126f70f462b2ee3f4bf711bdbd84b078d43c10e33` 与 `a88b4ca674fc3e122f48ba798d7898af2016e02ad4e24e6328f405c62a369007`。SR 重跑的预测 JSONL SHA256 两次均为 `00d626b4635f7473cb377ccbd5d66995914e53158c0d6ec5b3dda073487939cc`。

269 项 pytest、Ruff 与 `uv lock --check` 通过；四个方法的 `PointResult`/`RangeResult` JSONL 均为 300 条、无解析错误，冻结 Pilot 逐例校验通过。P5 在此停止；P6 才实现独立视觉方法。上述选择与分数只描述 Development Pilot，不推论真实 GNSS 异常或滑坡预警能力。
