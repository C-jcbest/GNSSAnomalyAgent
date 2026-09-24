# 数据契约与可复现流程

> 版本：2026-09-24 · 本篇描述 `normal-v1` 的磁盘文件、seed 树和本地 API。旧 180 日产物不在新接口中兼容或迁移。

## 输入与真值分层

每次请求创建一个数据集目录：

```text
data/generated/normal-v1-<timestamp>-<suffix>/
├── manifest.json
└── cases/
    └── case_0001/
        ├── input.json
        └── truth.json
```

`manifest.json` 记录 `generator_version="normal-v1"`、请求的 seed/数量、各案例 seed、进度和状态。`input.json` 是未来检测器允许读取的唯一案例文件。`truth.json` 是模拟器教学与事后评价用的独立文件；正式检测不得读取其背景、噪声或相位。所有生成数据默认被 `.gitignore` 排除。

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
| 真值 | `events` | P1 恒为空列表 |

H 和 R3D 从观测值确定性计算，不是额外传感器。P1 不保存 `true_coordinate`、`active_motion`、`persistent_offset` 或全零的事件贡献数组；P2 定义事件时再增加 `injected_deformation_mm` 与 `observation_artifact_mm` 等真值。

## 分层 seed 的复现边界

主 seed 是 $0$ 到 $2^{32}-1$ 的整数。第 $i$ 个案例（$i$ 从 0 开始）由 NumPy `SeedSequence([dataset_seed, i])` 产生一个 32 位 `case_seed`。再从 `SeedSequence(case_seed)` 派生三个独立子流：

```text
dataset_seed
  └── case_seed
        ├── annual_phase_seed
        ├── semiannual_phase_seed
        └── white_noise_seed
```

三个整数子 seed 保存在 `component_seeds`，JavaScript 可精确读取。改变白噪声的抽样流程不会隐式改变两组相位；未来 P2 可沿用相同案例 seed，令 paired cases 的正常背景保持一致，再另派生 deformation/artifact seed。**相同数据集 seed、案例序号和 `generator_version`** 应产生相同的输入与真值内容。批次 ID、创建时间不同，因此整个 manifest 不逐字相同。

## API 与生成流程

`POST /api/datasets` 的请求体严格只有两个字段，例如：

```json
{"seed":42,"count":20}
```

`count` 当前为 1～75。旧 `preset`、`config`、`days`、AR 参数等字段会返回 422；没有迁移层。服务先记录 `queued` manifest，后台逐例生成并更新进度，最终状态为 `complete` 或 `failed`。

- `GET /api/datasets`：列出当前 `normal-v1` 批次。
- `GET /api/datasets/{id}`：读取进度、版本及案例清单。
- `GET /api/datasets/{id}/cases/{case_id}`：读取检测输入。
- `GET /api/datasets/{id}/cases/{case_id}/truth`：仅供教学与评价读取真值。

CLI 使用同一逻辑：`uv run gnss-sim generate --seed 42 --count 20`。网页只允许修改 seed 和数量；固定模型参数以只读卡片展示。生成过程不访问平台或模型 API。

## 版本与旧产物

旧 `sim-*` 180 日目录不进入新批次列表，也不能经新 API 读取。新目录名与 manifest 明示 `normal-v1`；数据格式分别为 `normal-input-v1`、`normal-truth-v1`、`normal-dataset-v1`。P2 若改变事件规则，应建立 `event-v1`，不要借旧字段维持表面兼容。自动测试覆盖日期、成分求和、H/R3D、seed 复现、真值隔离及旧请求拒绝；工程自检不等于检测性能结论。
