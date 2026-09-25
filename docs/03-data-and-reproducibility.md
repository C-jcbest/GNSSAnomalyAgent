# 数据契约与可复现流程

> 版本：2026-09-25 · 本篇描述 P2/P3 `event-v6` 的临时批次文件、seed 树和本地 API。P4 固定数据另见 [Pilot 协议](05-p4-pilot-evaluator.md)。

## 输入与真值分层

每次请求创建一个数据集目录：

```text
data/generated/event-v6-<timestamp>-<suffix>/
├── manifest.json
└── cases/
    └── case_0001/
        ├── input.json
        └── truth.json
```

`manifest.json` 记录 `generator_version="event-v6"`、请求的 seed/数量/案例或场景类型、计划数量 `type_counts`、各案例的类型/seed/事件数、进度与状态。P2 案例至多一个事件；P3 场景案例有 2～6 个。`input.json` 是未来检测器允许读取的唯一案例文件。`truth.json` 是研究人员检查与事后评价用的独立文件；正式检测不得读取其场景、背景、噪声、事件或相位。所有生成数据默认被 `.gitignore` 排除。

## 字段与数学含义

| 文件 | 字段 | 含义 |
| --- | --- | --- |
| 输入 | `dates` | 2025-01-01 到 2025-12-31 的 365 个连续日期 |
| 输入 | `reference_coordinate_mm` | 固定 $P_0=(0,0,0)$ mm，顺序 N/E/U |
| 输入 | `observed_coordinate_mm` | 最终观测 $O_t$ |
| 输入 | `displacement_mm` | 带符号的 $\Delta_t=O_t-P_0$ |
| 输入 | `horizontal_offset_mm` | 水平偏移模长 $H_t$ |
| 输入 | `spatial_offset_mm` | 三维偏移模长 $R_{3D,t}$ |
| 真值 | `normal_background_mm` | annual + semiannual 的 $B_t$ |
| 真值 | `measurement_noise_mm` | 独立白噪声 $\epsilon_t$ |
| 真值 | `annual_phase_rad` / `semiannual_phase_rad` | 本例三轴的两组实际相位 |
| 真值 | `component_seeds` | 三路成分 seed |
| 真值 | `event_seeds` | 事件位置、形态、符号的独立子 seed；normal 为 null |
| 真值 | `injected_deformation_mm` | Step、Slow Trend、Acceleration 的逐日三轴贡献 |
| 真值 | `observation_artifact_mm` | Spike、Transient Shift 的逐日三轴贡献 |
| 真值 | `scenario_type` | P3 六场景之一；P2/normal 为 null，不进入检测输入 |
| 真值 | `events` | 不设 schema 数量上限的严格 typed event 列表 |
| 真值 | `event_contributions` | 按事件 ID 对应的独立 365×3 贡献及注入分量 |

H 和 R3D 从观测值确定性计算，不是额外传感器。每例均保存形变与伪差聚合数组，包括 normal 的全零数组；P3 聚合数组分别是对应逐事件数组之和，再逐元素核对 $O=P_0+B+\epsilon+D^{\mathrm{def}}+A^{\mathrm{art}}$。不保存含糊的 `true_coordinate` 或冗余的逐日状态。事件的起止索引从 0 开始、含两端，日期与索引必须一致；Step 的 `end_index=start_index` 只描述变化动作，不表示之后的偏移消失。

## 分层 seed 的复现边界

主 seed 是 $0$ 到 $2^{32}-1$ 的整数。第 $i$ 个案例（$i$ 从 0 开始）由 NumPy `SeedSequence([dataset_seed, i])` 产生一个 32 位 `case_seed`。再从 `SeedSequence(case_seed)` 派生三个独立子流：

```text
dataset_seed
  └── case_seed
        ├── annual_phase_seed
        ├── semiannual_phase_seed
        ├── white_noise_seed
        ├── event_position_seed  ┐
        ├── event_shape_seed     ├ 独立事件命名空间
        └── event_sign_seed      ┘
```

前三个整数子 seed 保存在 `component_seeds`。事件三路子 seed 由独立的 `SeedSequence([case_seed, 0x45564E54])` 派生并保存在 `event_seeds`。P3 的 shape 流选场景内数量、长期形态与轴，position 流选满足间隔的日期，sign 流选方向；这些抽样不消耗背景或白噪声流。相同案例 seed 的 normal、P2 与 P3 版本，其背景和噪声数组逐值相同。**相同数据集 seed、案例序号、类型和 `generator_version`** 产生相同的输入与真值内容。批次 ID、创建时间不同，因此整个 manifest 不逐字相同。

`case_type=all` 时，六类的目标比例依次为正常 25%，Spike、Step、Slow Trend、Acceleration、Transient Shift 各 15%。至少生成 6 例。设总数为 $n$、类型目标比例为 $p_k$，先置 $n_k=\max(1,\lfloor np_k\rfloor)$，再将剩余名额按 $np_k-n_k$ 从大到小分配；余数相同由独立的 `SeedSequence([dataset_seed, 0x4D4958])` 决定先后，最后使用同一独立流打散案例顺序。这个批次分配流不影响各案例的背景、噪声或事件 seed。例如 20 例恰好为正常 5 例、五类异常各 3 例。比例只是 P2 检查工作台的便利设置，不代表自然发生率或 P4 Pilot 配额。

`case_type=all_scenarios` 时按同一确定性余数规则在六种 P3 场景间**均衡**分配，至少 6 例；54 例每场景 9 例。这只是生成器验收集合，不代表自然发生率。也可直接指定一个场景生成同类案例。

P4 不使用上述网页批次。`uv run gnss-sim pilot --seed 20260925` 在 `data/pilots/pilot-v1/` 固定 300 例、每例 `input.json`/`truth.json` 分层存放、manifest 与 summary 单独保存。分层 seed 只针对预定的类型、轴、符号和场景；输出哈希及真值核查见 [P4 协议](05-p4-pilot-evaluator.md)。

## API 与生成流程

`POST /api/datasets` 的请求体严格只有三个字段，例如：

```json
{"seed":42,"count":20,"case_type":"all"}
```

`case_type` 必填，可选 `all`、`normal`、五个 P2 类型、`all_scenarios` 或六个 P3 场景名（见[场景协议](04-planned-methods.md)）。单类型/场景请求的 `count` 为 1～5000，两个混合选项为 6～5000；5000 是防误操作的**工程上限**，不是 Pilot 科学参数。旧两字段请求及 `preset`、`config`、`days`、AR 参数等字段返回 422；没有迁移层。服务先记录 `queued` manifest 和计划类型数量，后台逐例生成并更新进度，最终状态为 `complete` 或 `failed`。

- `GET /api/datasets`：列出当前 `event-v6` 批次。
- `GET /api/datasets/{id}`：读取进度、版本及案例清单。
- `GET /api/datasets/{id}/cases/{case_id}`：读取检测输入。
- `GET /api/datasets/{id}/cases/{case_id}/truth`：仅供教学与评价读取真值。

CLI 使用同一逻辑：`uv run gnss-sim generate --seed 42 --count 54 --case-type all_scenarios`。网页允许选择案例或场景类型、seed 和数量；固定模型参数以只读区域展示。左侧每次生成历史对应一个批次，案例按 manifest 中的类型折叠。研究人员选择案例后，网页分别请求观测输入和独立真值，默认展示标注；事件时间线支持按轴/事件族筛选、点选高亮与独立贡献查看。此交互图不作为正式视觉模型输入，未来检测器不能持有真值存储。生成过程不访问平台或模型 API。

## 版本与旧产物

新目录名与 manifest 明示 `event-v6`；数据格式分别为 `event-input-v6`、`event-truth-v6`、`event-dataset-v6`。旧版本目录即使仍在本地，也不由新 API 读取；不编写迁移。自动测试覆盖日期、五种纯事件 profile、六场景约束、逐事件及聚合贡献、逐元素组合、H/R3D、配对背景、混合分配和真值隔离；工程自检不等于检测性能结论。
