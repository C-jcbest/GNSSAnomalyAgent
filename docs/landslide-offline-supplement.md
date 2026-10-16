# 主结果离线评分复现包

日期：2026-10-10。对应全文 v23 的同次数位置确认批 case0121—0132 及追加 TAMA 设计适配，共 41 组、12 条受控记录。它补充已有逐实验复现命令，不改变方法、参考、预测或评分；新增模型请求为 0。

可交付压缩包：`artifacts/landslide-adaptive-review-2026-10-09/offline-supplement-delivery-v1/gnss-offline-supplement-v1.zip`，672843 bytes。SHA256：`6188887e16ad4e67da4ea770c248aa58b426b160835557d01a11a5f84f963769`。

## 运行

解压后进入 `gnss-offline-supplement-v1`，使用 Python 3.11：

```shell
python -m pip install -r requirements.txt
python replay.py --out ../replayed-results
```

输出目录必须不存在。复现运行无需原仓库目录、API 凭据或联网推理；安装依赖可能需要网络。实际验证使用已有独立复现环境，未声称在新机器重新安装依赖已通过。

包内保留原始三分量输入、机制参考、所有冻结预测、两项实验协议、家族元数据、v23 内容快照、比较图和原评分源码来源。`boundary_errors`、`evaluate_population`、`verify_metrics` 按 AST 提取完整函数，函数文本与包内原文件逐字一致；核心评价模块直接复制，既有源码未修改。各文件、函数和来源哈希可核查。包内没有凭据或请求传输实现作为运行依赖。

## 实际重算与验证

首先在原位置执行，再将压缩包解压到仓库外的系统临时目录，以该目录为 cwd，移除 PYTHONPATH，并用 Python `-E` 忽略环境路径变量运行。两次均得到：

- 全部 41 组的活动 Precision/Recall/F1、弱召回、未知覆盖、IoU .3/.5 事件与匹配起止边界。
- 端到端 A/S/D 和保留漏检／未知的真活动内部 A/S/D；原 sklearn 核验函数未改。
- 246 家族结果、72 条纠错记录、492 条逐记录摘要及 359160 行完整逐日输出。
- 完整 results/internal-stages 与原归档结构化结果相同；summary/per-case/family/corrections/daily CSV 与原 analysis-v3 逐字节相同，仓库外输出与原位置输出也逐字节相同。

实际更改解压测试副本的 reference.csv 后，重放在创建结果目录前拒绝运行；正式压缩包和冻结数据未修改。54 个冻结文件均在包内。Ruff 两个新增脚本通过，可读性复查完成。验证回执及 stdout/stderr 位于 `offline-supplement-delivery-v1`。

首次尝试使用 `-I`，其隔离模式排除了脚本目录，导致本地 `scoring` 模块找不到；未产生评分输出。改用 `-E` 后原代码成功，不为此引入路径回退或修改冻结源码；失败与重试方式已记录。构建前 Ruff 仅修复导入后空行，构建后源码不再改动。

## 复现承诺的范围

这是**冻结预测的评分复现**。成本列来自归档部署统计，不能称本次重新测得推理成本；本包不重新生成随机数据、拟合九个数值基线或调用 VLM，也不覆盖全部历史批次。完整训练、推理、协议冻结和请求来源仍以原仓库相关结果报告为准。重新调用远端模型不能承诺相同回答。

v23 快照保留原仓库相对文档／图片链接，未改其字节；这些历史链接不会在紧凑包中全部解析。包内比较图单独位于 `figures/comparison.png`。当前可浏览的完整稿和全部补充报告仍在原仓库。

参考是受控潜在运动机制标签，尚无正式现场独立复核参考。本包增加可复现交付能力，不增加识别效果证据，不改变强 XGBoost 更高的日 F1、TAMA 误报／覆盖优势、范围增益未完整复制或半合成背景失败。旧内部阶段核验器的 scope 文本仍写 selected targets，实际调用为全部 730 日参考活动；保持源文本与原输出一致，在此明确其真实范围。

## 构建入口

```powershell
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/build_landslide_offline_supplement.py --out artifacts/landslide-adaptive-review-2026-10-09/offline-supplement-new
```

构建目录必须新建。构建脚本检查原 analysis-v3 与 manuscript-v23 冻结文件，输出可直接执行的目录；交付 ZIP 只包含 manifest 所列文件及 manifest 本身，排除运行时 pycache。压缩包仓库外运行已验证，正式现场参考和作者审核仍待完成，goal 保持 active。
