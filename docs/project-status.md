# 项目状态与关键决定

## 2026-09-24 · P2 单轴单事件协议验收

- 当前生成器为 `event-v2`，旧 `normal-v1`、`event-v1` 数据和 API 契约均不兼容，也不迁移。2025 年 365 日、每日一值，固定 $P_0=(0,0,0)$ mm；annual 幅值 N/E/U 为 1.5/1.5/2.0 mm，semiannual 为 0.5/0.5/1.0 mm，白噪声标准差为 0.75/0.75/1.5 mm。三轴两组相位由独立 seed 抽样。数值是改善异常可辨识性的受控 benchmark 设定，并非现场精度或参考论文的完整参数。
- Normal 没有事件、线性速度、AR(1) 或 flicker noise。P2 异常例恰好一个事件和一个 N/E/U 轴：Spike 为单日观测伪差 $\pm6\sigma_c$；Step 为永久形变 offset $\pm5\sigma_c$；Slow Trend 为 90 日线性累计 $\pm6\sigma_c$；Acceleration 为独立的 90 日归一化二次累计 $\pm6\sigma_c$；Transient Shift 为 14 日矩形观测伪差 $\pm5\sigma_c$。事件不早于索引 60 开始，且不晚于索引 304 结束。正式公式见 [P2 协议](04-planned-methods.md)。
- 输入和真值分别保存。事件使用 typed truth，含类型、来源、轴、起止索引/日期、持续性、幅值/终值及时长；真值另存 deformation/artifact 数组。网页初始只读 input，研究人员主动点击后才读 truth；带真值的交互图只用于检查，不是未来 Visual benchmark 输入。
- 验收：49 项 pytest、Ruff、前端构建通过。以数据集 seed `20260924` 生成 normal 和五类异常各 30 例，共 180 例；逐元素组合等式成立，六种变体相同 case seed 的 background/noise 数组逐值完全一致。每类选 E/U/N 各一例，共 15 例，人工核对了贡献在起点、终点、终点次日和年末的值，并在网页逐例核对类型、轴和 Day 区间。桌面/手机工作台、事件详情、成分切换与文档公式也已检查。
- P2 不含多事件、多轴事件、缺测、检测器、模型 API、评价指标或 Agent。下一阶段须另行指令；当前工程自检不构成检测性能或真实滑坡预警结论。
