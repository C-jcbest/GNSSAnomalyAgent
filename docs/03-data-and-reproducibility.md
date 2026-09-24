# 数据契约与可复现流程

> 版本：2026-09-24 · 本篇描述 P2 `event-v3` 的磁盘文件、seed 树和本地 API。旧数据契约不在新接口中兼容或迁移。

## 输入与真值分层

每次请求创建一个数据集目录：

```text
data/generated/event-v3-<timestamp>-<suffix>/
├── manifest.json
└── cases/
    └── case_0001/
        ├── input.json
        └── truth.json
```

`manifest.json` 记录 `generator_version="event-v3"`、请求的 seed/数量/案例类型、各案例 seed 和事件数、进度与状态。`input.json` 是未来检测器允许读取的唯一案例文件。`truth.json` 是模拟器教学与事后评价用的独立文件；正式检测不得读取其背景、噪声、事件或相位。所有生成数据默认被 `.gitignore` 排除。

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
| 真值 | `events` | 严格 typed event；normal 为空，异常例恰好 1 个 |

H 和 R3D 从观测值确定性计算，不是额外传感器。P2 的每例均保存两个贡献数组，包括 normal 的全零数组，以便逐元素核对 $O=P_0+B+\epsilon+D^{\mathrm{def}}+A^{\mathrm{art}}$。不保存含糊的 `true_coordinate` 或冗余的逐日 `active_motion`、`persistent_offset`。事件的起止索引从 0 开始、含两端，日期与索引必须一致；Step 的 `end_index=start_index` 只描述变化动作，不表示之后的偏移消失。

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

前三个整数子 seed 保存在 `component_seeds`。事件三路子 seed 由独立的 `SeedSequence([case_seed, 0x45564E54])` 派生并保存在 `event_seeds`。相同案例 seed 的 normal 与任一事件版本，其背景和噪声数组逐值相同。**相同数据集 seed、案例序号、案例类型和 `generator_version`** 产生相同的输入与真值内容。批次 ID、创建时间不同，因此整个 manifest 不逐字相同。

## API 与生成流程

`POST /api/datasets` 的请求体严格只有三个字段，例如：

```json
{"seed":42,"count":20,"case_type":"slow_trend"}
```

`case_type` 必填，可选 `normal`、`spike`、`step`、`slow_trend`、`acceleration`、`transient_shift`。`count` 的 1～5000 是防止误操作的**工程上限**，不是 Pilot 科学参数。旧两字段请求及 `preset`、`config`、`days`、AR 参数等字段返回 422；没有迁移层。服务先记录 `queued` manifest，后台逐例生成并更新进度，最终状态为 `complete` 或 `failed`。

- `GET /api/datasets`：列出当前 `event-v3` 批次。
- `GET /api/datasets/{id}`：读取进度、版本及案例清单。
- `GET /api/datasets/{id}/cases/{case_id}`：读取检测输入。
- `GET /api/datasets/{id}/cases/{case_id}/truth`：仅供教学与评价读取真值。

CLI 使用同一逻辑：`uv run gnss-sim generate --seed 42 --count 20 --case-type slow_trend`。网页允许选择案例类型、seed 和数量；固定模型参数以只读卡片展示。网页先获取案例 `input.json`；研究人员点击「显示真值」后才请求 `/truth`，事件带和生成成分只在该状态展示。生成过程不访问平台或模型 API。

## 版本与旧产物

旧 `sim-*`、`normal-v1-*`、`event-v1-*` 与 `event-v2-*` 目录不进入新批次列表，也不能经新 API 读取。新目录名与 manifest 明示 `event-v3`；数据格式分别为 `event-input-v3`、`event-truth-v3`、`event-dataset-v3`。自动测试覆盖日期、五种事件贡献、逐元素组合、H/R3D、配对背景、真值隔离及旧请求拒绝；工程自检不等于检测性能结论。
