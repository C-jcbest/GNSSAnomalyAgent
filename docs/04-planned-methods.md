# P2 单事件生成与后续检测草案

> 版本：2026-09-24 · **P2 的五种单轴单事件生成规则已实现为 `event-v2`。** 本篇从“多事件与多标签真值”开始的 P3～P9 内容仍是草案；生成正确性不能当成检测性能结论。

## P2 的目标与边界

P2 每例选择 `normal` 或五种异常之一。异常例恰好一个事件、一个轴；正常例没有事件。没有多事件、跨轴注入、缺测、难度矩阵、检测器、模型调用或 Agent。索引 $t,s,e$ 从 **0** 开始，区间含两端；`Day 1` 对应索引 0。

在 `event-v2` 的固定正常背景和噪声上，逐元素生成

$$
O_{t,c}=P_{0,c}+B_{t,c}+\epsilon_{t,c}+D^{\mathrm{def}}_{t,c}+A^{\mathrm{art}}_{t,c},
\qquad 0\le t<365,\quad c\in\{N,E,U\}.
$$

真值分别保存 `normal_background_mm`、`measurement_noise_mm`、`injected_deformation_mm` 和 `observation_artifact_mm`。正常例的后两项为全零。模拟器的 `source` 只表示注入数组，不是检测器必须判断的现场成因；检测器将来只输出可观察的形态、轴和日期。Step、Slow Trend 和 Acceleration 注入形变；Spike 与 Transient Shift 注入观测伪差。

## 五种已冻结的事件形态

下表仅写事件轴 $c$ 的非零贡献；另外两轴始终为零。$\sigma_N=\sigma_E=0.75$ mm、$\sigma_U=1.5$ mm，正负号由独立 `event_sign_seed` 决定。$s$ 是起点、$e$ 是终点。

| 类型 | 非零贡献 | P2 固定参数 | `end_index` 与偏移语义 |
| --- | --- | --- | --- |
| Spike | $A^{\mathrm{art}}_{t,c}=A\mathbf1[t=s]$ | 1 日，$\lvert A\rvert=6\sigma_c$ | $e=s$；下一日恢复，`persistent=false` |
| Step | $D^{\mathrm{def}}_{t,c}=A\mathbf1[t\ge s]$ | 起变 1 日，$\lvert A\rvert=5\sigma_c$ | $e=s$ 是**变化动作**的终点；此后保留偏移，`persistent=true` |
| Slow Trend | $D^{\mathrm{def}}_{t,c}=M\,\operatorname{clip}((t-s)/89,0,1)$ | 90 日、带符号最终偏移 $\lvert M\rvert=6\sigma_c$ | $e=s+89$；之后保留 $M$，斜率 $v=M/89$ mm/日 |
| Acceleration | $D^{\mathrm{def}}_{t,c}=M\,[\operatorname{clip}((t-s)/89,0,1)]^2$ | 同为 90 日、$\lvert M\rvert=6\sigma_c$ | $e=s+89$；独立凸形变，之后保留 $M$ |
| Transient Shift | $A^{\mathrm{art}}_{t,c}=A\mathbf1[s\le t\le e]$ | 14 日、$\lvert A\rvert=5\sigma_c$ | $e=s+13$；$e+1$ 日严格归零，`persistent=false` |

`clip(x,0,1)` 将 $x$ 限制在 0 与 1。Slow Trend 与 Acceleration 具有相同的持续时间和最终偏移，唯一差异是区间内的线性与凸形轨迹。这里的 5σ、6σ 是便于核对生成正确性的 canonical 设定，**不是**现场异常强度或典型滑坡速度。Point/range 形态的研究动机可参见 [VisualTimeAnomaly](https://github.com/mllm-ts/VisualTimeAnomaly)；本项目固定幅值是自己的受控 benchmark 设定，未复用其生成代码或宣称现场真实性。

对应的绝对幅值为：Spike、Slow Trend 终值和 Acceleration 终值在 N/E 为 4.5 mm、U 为 9.0 mm；Step 与 Transient Shift 在 N/E 为 3.75 mm、U 为 7.5 mm。符号可正可负，Slow Trend 与 Acceleration 的 `final_offset_mm` 保存带符号终值。

P2 的事件起点满足 $s\ge60$、终点满足 $e\le304$，即至少保留前后各 60 日上下文。Step/Spike 的 `end_index=s` 表示瞬时变化日期；**不把**后续稳定偏移误标为持续运动。Slow Trend/Acceleration 的活动区间为 $[s,e]$，其后可由 `persistent=true` 与事件参数推导残余偏移，不另外存储逐日状态数组。

## 事件真值与随机数

真值 `events` 使用严格的类型联合，每例长度为 0 或 1。事件统一含 `event_id`、`type`、`source`、`axis`、起止索引/日期和 `persistent`；每类 `parameters` 有各自的类型约束。`event_001` 是 P2 唯一事件 ID。日期必须与 2025 年索引一致。

`case_seed` 的 annual 相位、semiannual 相位和白噪声三路 seed 与旧 P1 派生方式相同。事件另用 `SeedSequence([case_seed, 0x45564E54])` 派生 `position`、`shape`、`sign` 子 seed，记录在真值中。相同 `case_seed` 的 normal/五种事件版本共享**完全相同**的背景与噪声；只有事件贡献改变。`shape` 子流选择轴，幅值由该轴固定的 $\sigma_c$ 决定；起点从满足安全区和事件时长的所有整数位置等概率抽取。

P2 只验证生成器：逐元素等式、独立数组、边界与日期、事件类型、配对背景，以及 N/E/U 的 15 种 canonical 组合。尚无检测运行或模型结果。

## 多事件与多标签真值

P3 计划组合已验证的纯事件贡献：$D^{\mathrm{def}}=\sum_j d^{(j)}$、$A^{\mathrm{art}}=\sum_k a^{(k)}$，再在 P4 构造按日、轴、类型的指示张量。令 $K$ 为类型集合，

$$
Y_{t,c,k}=\mathbf 1\big[\text{日期 }t\text{ 在分量 }c\text{ 上属于类型 }k\text{ 的事件}\big],
\qquad Y\in\{0,1\}^{365\times3\times |K|}.
$$

因此一次趋势中的尖峰可以同时拥有 Slow Trend 与 Spike 标签；趋势后期再加速也能同时拥有 Slow Trend 与 Acceleration 标签。届时需要明确活动区间与残余偏移的评价语义，避免把已稳定的偏移误称为持续运动；是否保存逐日状态由评价需求决定。缺测条件不属于 P2。

## 数值与视觉方法如何比较

下列都是**候选基线**，没有写入当前运行入口：

| 候选方法 | 主要观察量 | 教学用简式 |
| --- | --- | --- |
| Hampel / 鲁棒 Z 分数 | 点相对局部中位数的偏离 | $z_t=\lvert x_t-\operatorname{med}(\mathcal W_t)\rvert/(1.4826\operatorname{MAD}(\mathcal W_t)+\epsilon)$ |
| PELT / 变化点方法 | 分段拟合代价的下降 | $\min_{m,\tau_1,\ldots,\tau_m}\left\{\sum_{j=0}^{m} C(x_{\tau_j:\tau_{j+1}})+\beta m\right\}$ |
| Theil–Sen 滑动斜率 | 局部长期方向与速度 | $\widehat v=\operatorname{med}_{i<j}\left\{(x_j-x_i)/(j-i)\right\}$ |
| 斜率变化 | 相邻窗口速度变化 | $\Delta\widehat v=\widehat v_{\mathrm{recent}}-\widehat v_{\mathrm{previous}}$ |

上式仅说明各方法测量什么。其中 $\mathcal W_t$ 是围绕 $t$ 的局部观测窗口；PELT 式中 $\tau_0=0$、$\tau_{m+1}=365$，切分边界采用左闭右开索引。窗口宽度、阈值、代价函数 $C$、惩罚系数 $\beta$、缺测策略和输出事件转换规则，都需要在对应阶段开发与验证，不能在看测试结果后再选。固定数值组合应事先确定如何合并候选，并保留来源。纯视觉方法拟查看同一输入生成的 N/E/U 带符号位移与 H 四联图；图中不得包含事件标签或注入参数。视觉模型应允许返回零个、一个或多个事件。365 日图是否压缩信息，待实验证据出现后再考虑窗口长度消融。

## 事件输出与评价

未来的统一预测项拟包含可观察形态 `type`、分量 `axes`、`start` 和 `end`；起止索引从 0 开始并含两端。事件级比较需先定义兼容条件，再做**一对一匹配**，以免一个宽预测重复抵消多个漏报。对区间事件，可讨论时间交并比

$$
\operatorname{IoU}(I,\widehat I)=
\frac{|I\cap\widehat I|}{|I\cup\widehat I|};
\qquad
\operatorname{onset\ error}=|s-\widehat s|.
$$

Spike 或 Step 的起点可用预先冻结的日期容差评价；Slow Trend 和 Acceleration 还需报告起止误差或时间重叠。窗口级、事件级、类型级与定位指标分别展示，不能只给一个总 F1。若某组没有阳性真值，阳性 F1 应记为不适用，另报告完成案例上的报警情况。具体匹配阈值和汇总口径将在 P4 评价器实现前确定。

## Pilot 与独立测试的顺序

P4 计划先生成约 300 个 Pilot 案例：正常 75、单异常 75、多异常 100、困难 50。四组计数互斥，困难属性仍可与多事件等特征交叉；随机参数由固定 seed 控制。先验证生成形态与标签，再使用人工构造的正确、漏报、误报、错位预测检验评价器。Pilot 属于开发验证，不等于独立最终测试。

之后数值、视觉和固定组合必须使用同一批检测输入，测试真值不得进入提示词、路由、参数选择或工具输出。只有在开发分析显示互补性时才设计 Agent；还需与较强的单方法和固定组合比较，并披露模型请求与总耗时。P2 尚无这些结果。
