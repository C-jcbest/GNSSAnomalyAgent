# GNSSAnomalyAgent

面向GNSS坐标序列的离线异常检测研究：定位N/E/U中的异常数据点和时间段，不凭曲线判定故障、形变或滑坡成因。固定采用北京时间每日15点、180天窗口的日观测，比较数值、视觉与固定组合，独立候选合并时去重并保留来源，不要求共同确认。只有后续能力证据支持时才另行考虑轻量Agent。

**当前阶段：固定15点的受控注入比较及原生缺测敏感性实验均已完成；现场准确率尚不可评估。** 初轮144例含额外缺测派生，后续仅保留36个原生缺测案例。后者是测试结果已知后的事后分析，不是新的独立盲测；F1仅为注入掩码一致度，不作为真实预警或Agent优势结论。详见[项目状态](docs/project-status.md)。

## 快速查看

Python 3.11+，在仓库根目录执行：

```powershell
uv sync
uv run gnss-exp serve
```

打开 <http://127.0.0.1:8765>，选择日注入运行的“注入标注”页，可按来源窗和缺测状态查看六类受控情景的三轴标注图；“案例证据”页另看检测预测。工作台只读，不触发平台采集或模型调用。没有uv时可用现有环境的 `.venv/Scripts/gnss-exp.exe serve`。

## 文档导航

| 要了解什么 | 文档 |
| --- | --- |
| 固定15点的数值/视觉/组合对照与执行边界 | [实验方案](docs/experiment-plan.md) |
| 已完成结果、限制与待办 | [项目状态](docs/project-status.md) |
| 采集、收束、分组与本地位置 | [数据准备](docs/data-preparation.md) |
| 每日15点、180天与SCWM-04口径 | [日数据契约](docs/daily15-data.md) |
| 命令、历史查看、导出与模块 | [运行手册](docs/implementation.md) |
| P/R/F1、缺测、失败和结论标准 | [指标标准](docs/metrics-criteria.md) |
| 日预实验与融合的冻结协议 | [日预实验协议](docs/daily-pilot-protocol.md) |
| 固定15点的新测试集和同期比较 | [日同期比较协议](docs/daily-comparison-protocol.md) |
| 不额外注入缺测的事后实验 | [原生缺测协议](docs/daily-native-only-protocol.md) |
| 低分来源与测试后诊断 | [日比较诊断](docs/daily-comparison-diagnosis.md) |
| reference_trend的参数与限制 | [长期趋势候选协议](docs/daily-trend-pilot.md) |
| AI审核入口与填写规则 | [背景审核](docs/background-review-guide.md) |
| TSAD-Agents、TSB-AD、TAMA等依据 | [文献调研](docs/literature-review.md) |
| GNSS日周期与日代表值依据 | [日周期专题](docs/gnss-daily-preprocessing-review.md) |

参数以对应冻结协议及配置为准，指标以指标标准为准，结果统一记录于项目状态。一次性模型对比的参数和源码包仅在本地`artifacts/model-migration-v{1,2}`追溯；旧小时方案不作为当前执行要求。

## 本地资料与Git

远端：`git@github.com:C-jcbest/GNSSAnomalyAgent.git`。代码、配置和文档用于Git管理；`.env`、原始数据、标签、模型响应和运行产物默认忽略。连接变量见 [.env.example](.env.example)。

`papers/local/`存论文草稿，`research/local/`存调研缓存和文档备份，均不提交或推送。草稿只作参考，不能直接复用其中结果。
