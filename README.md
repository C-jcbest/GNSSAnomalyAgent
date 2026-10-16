# GNSS 模拟实验台

最新研究状态：[完整初稿v14](docs/gnss-deformation-manuscript-draft-2026-10-13-v14.md)与[QSG完整结果](artifacts/gnss-deformation-diversity-2026-10-12/query-shape-grounding-report-v1/report.md)。局部图形监督F1 .7114低于去监督.7443、强数值.8442，净−68日，固定版停止；14模型266配置600输出重放、102公式通过。96×730、48／24／24不变，保留内容未加载；稳定Agent/VLM贡献尚未成立。

比较数值与视觉方法对合成日尺度 N/E/U 异常的识别能力。生成器复用 GutenTAG 和 TODS，支持 Normal、Global Extremum、Trend、Transient Mean Shift。

另提供长期滑坡位移仿真：同一记录内组合静稳、缓慢形变、加速、减速、停滞与再活动，允许中途起变和首尾截断。支持时间相关误差、缺测、批量曲线和观测派生速度/加速度辅助图；这些数据是文献机制启发的仿真，不是现场实测。

- [原方法新种子冻结确认](docs/landslide-cross-seed-confirmation-results.md)：整体视觉收益复制，范围严格条件未全部复制；117请求、29配置、聚焦全文v20。
- [正常实例检索与数值初判补充](docs/landslide-normal-anchor-results.md)：检索两遍失败，数值初判首遍提前停止；72新请求、42评分配置行，论文v19。
- [新增360日上下文与重复图成对取证](docs/landslide-context-observation-results.md)：68真实请求、24项强基线与消融，未稳定超原局部复核，停止该版本，论文v18。
- [四背景部署提案校准与完整迁移](docs/landslide-grouped-proposal-results.md)：真实165请求、23/28配置；验证通过但完整无稳定超数值投票增益，停止分支，论文v17。
- [实际视觉纠错收益训练与完整迁移](docs/landslide-review-utility-results.md)：小规模通过、完整24条未稳定提高；提案接受与简单数值投票等价，均停止。288新请求、完整消融和论文v16。
- [历史方向约束与前缀锚点](docs/landslide-direction-review-results.md)：两版主组无识别增量且覆盖下降，停止；同信息数值控制及全部消融，完整稿v15，新VLM请求0。
- [历史尺度压缩的观察准入](docs/landslide-scale-review-results.md)：开发弱尾收益与调用减少，第二遍事件条件未通过、位置选择优势未建立；28配置及同信息学习控制，全文v14。
- [单向漏运动修订三项开发假设](docs/landslide-missed-motion-repair-results.md)：逐日、校准与连续性三版无稳定收益，停止扩展；129评分行、九强控制、完整图表与论文v13，新增API请求0。

## 启动

需要 Python 3.11+、uv、Node.js 20+。在项目根目录运行：

```powershell
uv sync
uv run python scripts/setup_generator.py
npm --prefix web ci
npm --prefix web run build
uv run gnss-sim serve
```

打开 <http://127.0.0.1:18765>。生成器使用独立的 `.venv-generator/` 环境。

```powershell
uv run gnss-sim generate --seed 20260930 --count 24 --case-type all
uv run gnss-sim generate-landslides --seed 20261009 --count 24 --days 1095
uv run gnss-sim --help
```

## 文档

- [当前实验全流程详解](docs/landslide-experiment-design-walkthrough.md)：数据准备、两类参考标签、配对训练、参数选择、对照与评价。
- [120 条全量对比结果](docs/landslide-persistence-scale-results.md)：19 个冻结方法、全部记录、两套参考、误报与边界代价、逐记录 CSV 和重放证据。
- [数据生成](docs/data-generation.md)：上游来源、异常参数和标签。
- [长期滑坡数据设计](docs/landslide-design.md)：调研依据、多年过程与展示口径；网页入口为“长期位移”。
- [检测与评价](docs/detection.md)：校准、输入隔离、模型预算和评分。
- [开发与运行](docs/development.md)：目录职责、命令和检查。
- [项目状态](docs/project-status.md)：当前有效范围。
- [整体GNSS形变异常研究](docs/gnss-deformation-anomaly-method-directions-2026-10-12.md)：当前研究入口；[最新完整初稿v7](docs/gnss-deformation-manuscript-draft-2026-10-13-v7.md)与[实际进展](docs/project-status.md)。运动分布数值信息改善48／24开发F1，但有序辅助视觉未超普通融合与同信息强数值，16／8、21方法及消融完整保存；24保留输入／参考未打开。
- [有限运动冲突视觉复核](docs/landslide-adaptive-review-method.md)：历史受控方法；[统一实验结果](docs/landslide-adaptive-review-results.md)与[历史完整初稿](docs/landslide-adaptive-review-manuscript-draft.md)。
- [原始速率矛盾阶段复核](docs/landslide-stage-contrast-review-results.md)：开发修正105/46阶段日；新12条冻结确认两遍0触发，停止该选择器扩展。两批各24配置、强基线与全部失败保留，全文v12。
- [同参照完整有限复核原型](docs/landslide-reference-review-results.md)：12完整记录、37配置、同信息强基线；未通过完整开发条件，停止该参照观察扩展，全文v11。
- [训练年参照与附加视觉观察](docs/landslide-reference-observation-results.md)：硬触发否定、软观察48请求开发通过；完整12条原型后续失败并停止，历史完整稿v10。
- [观测残差背景稳健性实验](docs/landslide-background-benchmark-results.md)：60条半合成、共同重拟合九强基线、24测试两遍29配置；弱尾有收益，整体误报增加，范围仅撤回1／2日，完整稿v9。
- [竞争运动描述开发试验](docs/landslide-motion-competition-results.md)：八已见目标、三组两遍48请求、15配置；恢复弱活动但误报增加，停止模块，完整稿v8。
- [官方三分量迁移诊断](docs/landslide-field-transfer-results.md)：真实输入、稳定参照和无标签风险；[最近邻代码核查](docs/landslide-nearest-method-code-review.md)。
- [整段机制与追加确认](docs/landslide-interval-consistency-results.md)：新12记录、实际TAMA的GNSS适配及严格过滤的失败结果。
- [冲突范围修订与新确认](docs/landslide-revision-scope-results.md)：保持局部回答、限定可修订日期；30配置统一比较和等日数消融。
- [84日证据开发检验](docs/landslide-longer-evidence-results.md)：视觉扩展否定、39组统一比较、同支撑强机器学习控制及完整稿v5。
- [证据选择显示新确认](docs/landslide-evidence-display-confirmation-results.md)：0085–0096、25配置、两遍180请求；显示分支0触发、未升级，强基线与全部取舍保留，完整稿v7。
- [证据与观察显示匹配](docs/landslide-evidence-observation-results.md)：上下文×块图配对、完整窗口对照、45配置及完整稿v6的开发依据。
- [协作规范](AGENTS.md)。

实验登记、LangChain 工具调用和对照命令见[开发与运行](docs/development.md)，当前结论与本地报告入口见[项目状态](docs/project-status.md)。
