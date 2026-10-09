# 论文代码复现与运行检查

日期：2026-10-07。已执行公开代码示例及 TAMA 开发试运行；不是原论文结果表的完整重现，也不是正式滑坡性能比较。源码来源核查见[前置报告](../artifacts/reproducibility-audit-2026-10-07/code-audit.md)。

## 已完成的运行

| 方法 | 本轮实际运行 | 结果 | 可支持的结论 |
| --- | --- | --- | --- |
| PELT / ruptures 1.1.10 | 线性趋势代价，与独立穷举的带惩罚分段比较；另查静稳与常数平移 | 两者均得到结束位置 12、24、36，目标值 1.5467676739；静稳不分裂；平移不改变分段 | 当前数值实现通过这些算法检查；不代表所有代价与数据均已验证 |
| XGBoost 3.0.2 | 固定 commit 中未修改的官方多分类 train.py，UCI dermatology 示例 | 256 条训练、110 条测试，准确率 0.9091；多数类基准 0.3182；保存重载预测完全一致 | 官方示例及序列化可运行；这是皮肤病分类示例，不是 GNSS 准确率 |
| TCN / torch 2.8.0+cpu | 未修改作者 Adding Problem 代码，CLI 设 3 epochs、长度 64、5 层、每层 8 通道；50,000 训练、1,000 测试 | MSE 0.00044954，常数 1 基准 0.16718；因果性与保存重载一致性检查通过 | 作者骨干能训练、测试；这是缩小配置，不是论文长序列成绩，也不是滑坡分割成绩 |
| TAMA 适配版 / qwen3.8-flash | 保留原 CLI 正常参考→检测→多尺度复核流程；两条已有 GNSS 开发记录的 N 轴＋一条独立静稳合成对照 | 两轮各 9 次调用；第一轮 2/3 最终记录越界，第二轮 3/3 通过严格区间契约 | API、图像、参考与复核流程可运行；有任务语义差异，尚不能直接评价阶段性能 |

TCN 原程序 `--cuda` 使用 store_false，该参数实际关闭 CUDA；本次选 CPU，未占用现有 GPU 工作负载。除了 torch seed，驱动脚本也固定了作者数据生成器使用的 NumPy seed。训练骨干的理论感受野覆盖本次 64 日序列。原代码的 weight_norm 弃用警告保留，未为消除警告改写作者实现。

完整结构化结果：[summary.json](../artifacts/reproduction-2026-10-07/summary.json)。运行日志、模型与逐请求记录在同目录 runs/。全环境锁位于 configs/reproduction-requirements.txt，源码文件 URL/SHA256 位于 configs/reproduction-sources.json。

验证结果：独立复现环境的 7 项新增检查通过；应用环境的完整测试为 46 passed、1 skipped（复现测试模块因该环境未装 scikit-learn 而明确跳过，已在独立环境执行）。本次新增脚本与测试的 Ruff 检查通过，15 个上游文件的 SHA256 在运行后复核一致。Matplotlib/PyTorch 的上游弃用警告不影响上述通过结果。

## TAMA 检查发现与处理

复用固定 commit `db99a75a83d0ef05caf7ff0d199b75b594e98584` 的 main_cli.py、ProcessedDataset 和基类。原始文件单独保存并核对 SHA256；只对 adapted 副本打补丁，保存 portability.patch。

1. **正常参考存在测试标签回退。** 本项目显式禁止；为每条开发输入提供相同三条独立生成的静稳训练参考。推理目录中的标签全为 -1 占位，未读取现有阶段标签或生成真值。
2. **原 remove_padding 会压缩内部缺测。** 局部放大图由此可能改变日期坐标。当前数据没有尾部 padding，适配器保留完整长度与内部 NaN；缺测仍显示断线，不补成观测。
3. **原提示将横纵轴文字写反。** 改为横轴时间、纵轴位移，与实际绘图一致。
4. **默认未启用复核。** 本次显式传入 --double_check，实际两轮均执行了参考、检测、复核共 9 次调用。
5. **边界与输出类型不稳定。** 第一轮 case_0003、case_0009 的终点均为 1100，而实际索引是 0–1094；后者初次还返回了数组而非原协议字符串。保留原输出并判为契约失败，没有静默裁剪。第二轮修正图的最后刻度、局部坐标范围，明确合法索引与输出类型，加入严格解析后全部通过。
6. **费用、重试与凭据。** 禁用原代码的额外 JSON 格式化模型和自动重试；每轮最多 9 请求、每次最多 4096 输出 token。使用已有本地模型配置，凭据不写日志。原脚本的固定单价估算被关闭，记录实际 token 而非假定费用。

两轮总调用 18 次，返回总 token 80,763；第一轮 39,849，第二轮 40,914。流程耗时约 59.2 秒、57.4 秒，包含当前服务条件，不能视为稳定吞吐承诺。第二轮静稳对照初次报 1025–1035 的 contextual 异常，复核后为空；这只是一个开发例，不能证明总体误报率改善。

第二轮同时改变了坐标/契约和随机参考顺序控制，模型调用也具有随机性；不能把两轮区间差异单独归因于某个修复。两轮均保留，不挑较好版本作为独立测试成绩。

## 从开发图中看到的任务差异

![TAMA 开发输出](../artifacts/reproduction-2026-10-07/tama-development.png)

橙色仅表示 TAMA 的通用异常预测，不是活动金标准，更不是阶段标注。case_0003 的 240–610 区间跨过中间平台，case_0009 同时报告孤立尖峰和长趋势，后段范围也包含位移变缓后的部分记录。这说明原始通用异常任务与“持续位移活动”不完全一致。

正式对比应保留作者流程，但向所有视觉方法提供同一活动定义；将原始通用提示结果作为次要迁移分析。不能用本文专门的活动标签去惩罚完全未收到活动定义的对照，再据此声称层级方法优越。

## 未完成的复现

- **AnomaMind：** 源码与发布权重可获取，但本轮没有运行其完整 Locator/Actor/Detector/Evaluator 链。现有模型服务不是作者发布的检测权重服务；仓库默认同时引入 vLLM、verl、flash-attn 等部署/训练依赖。本机是 Windows、32 GB 级内存、RTX 4070 Laptop 8 GB，检查时空闲显存约 3.7 GB。8B 原配置不是当前环境直接可启动的任务；这不等于断言 CPU/offload 或独立 Linux 部署不可行。下一步先核对独立部署配置，不用通用模型偷偷替代作者检测器。
- **完整 TSAD-Agents：** 已核实的仓库仍未提供可确认的完整执行链；直接视觉入口不能冒称其完整复现。
- **Refining：** 尚未找到可确认的官方代码，未实现或运行。
- **滑坡端到端比较：** XGBoost/TCN 尚未在独立滑坡标签上训练、PELT 尚未接统一阶段评价器。本轮不填论文效果表。

## 可重复运行的命令

以下安装命令只使用独立环境，不改应用依赖。已有输出目录会拒绝覆盖；新运行请换 --out 路径。模型命令会使用本机 .env 并产生外部调用。

```powershell
uv venv --python 3.11.11 artifacts/reproduction-2026-10-07/environment
uv pip sync --python artifacts/reproduction-2026-10-07/environment/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cpu configs/reproduction-requirements.txt
.venv/Scripts/python.exe scripts/setup_reproduction.py
artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/check_reproduction.py pelt --out artifacts/reproduction-2026-10-07/runs/pelt-new
artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/check_reproduction.py xgboost --out artifacts/reproduction-2026-10-07/runs/xgboost-new
artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/check_reproduction.py tcn --out artifacts/reproduction-2026-10-07/runs/tcn-new
artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/reproduce_tama.py prepare --out artifacts/reproduction-2026-10-07/runs/tama-new
artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/reproduce_tama.py execute --out artifacts/reproduction-2026-10-07/runs/tama-new
artifacts/reproduction-2026-10-07/environment/Scripts/python.exe -m pytest tests/test_reproduction.py -q
```

测试当前针对记录中的 tama-pilot-v2 构建产物；在全新机器上先以该目录完成 prepare。缺少独立依赖或上游文件时测试明确 skip，不声称已验证。已通过的 7 项检查覆盖测试标签隔离、缺测坐标、轴说明、持久请求预算、关键上游函数保留与严格区间解析。
