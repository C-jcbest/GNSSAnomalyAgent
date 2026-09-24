# 数据契约与可复现流程

> 版本：2026-09-24 · 描述当前 P1 的本地文件与 API。此文档不把计划中的异常标签当成现有产物。

## 三层数据与真值隔离

一次生成请求形成一个数据集目录：

```text
data/generated/<dataset_id>/
├── manifest.json
└── cases/
    └── case_0001/
        ├── input.json
        └── truth.json
```

`manifest.json` 记录请求配置、配置哈希、每例派生 seed、生成进度和完成/失败状态；`input.json` 是未来检测器可读的案例输入；`truth.json` 保存模拟器已知的背景、噪声、无噪声坐标、事件和状态。数据与真值默认被 `.gitignore` 排除，不进入源码提交。

## 字段如何对应公式

| 文件 | 字段 | 数学含义 / 说明 |
| --- | --- | --- |
| 输入 | `dates` | 180 个连续日日期，索引从 0 开始 |
| 输入 | `reference_coordinate_mm` | 固定 $P_0$，顺序恒为 N/E/U |
| 输入 | `observed_coordinate_mm` | 最终观测 $X_t$ |
| 输入 | `displacement_mm` | 带符号的 $\Delta_t=X_t-P_0$ |
| 输入 | `horizontal_offset_mm` | 水平模长 $H_t$ |
| 输入 | `spatial_offset_mm` | 三维模长 $D_t$ |
| 真值 | `background_displacement_mm` | 正常背景 $B_t$ |
| 真值 | `white_noise_mm` / `ar_noise_mm` | $W_t$ / $R_t$ |
| 真值 | `observation_noise_mm` | $W_t+R_t$ |
| 真值 | `true_coordinate_mm` | 模拟器的无噪声坐标 $T_t=P_0+B_t$ |
| 真值 | `events`、`active_motion`、`persistent_offset` | P1 分别为空列表、全 false、全 false |

`input.json` 中的 H 和 D 是观测值的确定性派生量，并非独立传感器数据。为了避免未来检测泄漏，检测入口应只读取 `input.json`；网页可以为教学展示另行读取 `truth.json`，不能将它回传给检测器。

## 随机种子与确定性

请求含一个主 seed，取值为 $0$ 到 $2^{32}-1$。第 $i$ 例（$i$ 从 0 开始）的独立整数 seed 由 NumPy 的 `SeedSequence([master_seed, i]).generate_state(1, dtype=np.uint64)[0]` 派生，再交给该例的随机数生成器。这避免在同一批次中按顺序复用单条随机流。

$$
s_i=\operatorname{generate\_state}_{\mathrm{uint64}}
\!\left(\operatorname{SeedSequence}([s_{\mathrm{master}},i]),1\right)_0,\qquad
(X_i,\mathrm{truth}_i)=G(s_i,\theta).
$$

这里 $\theta$ 是完整生成配置。相同主 seed、配置、案例序号和代码版本应得到相同的输入与真值内容。`dataset_id` 含创建时间与随机后缀，`created_at` 也会变化，因此**批次元数据本身不逐字相同**。`config_sha256` 是请求 JSON 的 SHA-256，用于识别配置；它不是案例内容哈希，也不能单独证明运行代码版本相同。

## 本地生成与读取流程

1. 发送 `POST /api/datasets`，请求包含 `preset=normal-p1`、seed、案例数和可选配置。案例数当前限制为 1～75。
2. 服务创建 `queued` 的 manifest；后台线程按案例生成输入与真值，逐例更新 `generated_cases`；最终变为 `complete` 或 `failed`。
3. `GET /api/datasets` 列出批次，`GET /api/datasets/{id}` 查看进度与案例清单。
4. `GET /api/datasets/{id}/cases/{case_id}` 读取检测输入；教学页面另用 `/truth` 端点读取真值。两者是独立接口。
5. 网页切换分量、缩放和日期，仅改变浏览视图，不改磁盘文件。

CLI 也能同步执行相同生成逻辑：`uv run gnss-sim generate --seed 20260923 --count 10`。生成过程不调用平台或模型 API。

## 校验与当前限制

自动测试检查同 seed 内容一致、180 个逐日日期、成分求和、固定参考坐标下的 H/D 公式、真值与输入分离及 API 进度。P1 不含异常，因此没有异常召回率、事件 F1 或检测运行结果。后续阶段新增字段和评价规则时，应同步更新本契约、生成代码、测试及网页说明。
