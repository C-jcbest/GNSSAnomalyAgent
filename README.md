# GNSS 视觉大模型异常识别研究

当前完成 **GNSS-VLM-EXP02-v3：GNSS活动与质量异常分离的时间解码及局部视觉复核实验**。
在`codex/gnss-vlm-research`分支新增邻日形态特征、训练侧时间解码、单轮局部原始NEU复核及按任务归属的融合，未设统一物理告警阈值。
三个新数据种子、72个独立测试站：主方法`anchored_fusion`形变日/事件F1为 **0.6038/0.4348**，
相同数值定位`rf_split_temporal`为 **0.5412/0.3509**，原方案`fusion`重新测试为 **0.3050/0.3562**。
自由视觉改写对照`guided_fusion`为 **0.6581/0.6667**；不把得分最高的对照事后改称主方法。

- [EXP02三个独立种子结果](results/exp02-three-seeds/results.md)
- [EXP02主种子对照、消融及复测](results/exp02-main-v3/results.md)
- [EXP02冻结方案与方法对应](docs/experiment-exp02.md)
- [开发负结果与选择依据](docs/exp02-development.md)
- [研究判断与下一步实质问题](docs/exp02-assessment.md)

主方法相对相同数值法的形变日F1差值95%区间为[0.0197,0.1278]，条件于三组已训练模型。
尚未稳定优于默认RF，局部图增量未证实，阶跃识别仍薄弱；三种子仿真不是现场验证。
实际模型请求尝试420次（含开发、失败、主批、独立种子和同站复测），派生融合结果不重复计费/计数。
23个测试及Ruff通过；主批/复测冻结源码匹配，24站RF确定性重算及24个复测请求哈希一致。

## EXP02运行

```powershell
uv sync --locked
uv run gnss-research generate --out data/reproduce-exp02 --seed 2026101203 --per-split 24
uv run gnss-research register --dataset data/reproduce-exp02 --out runs/reproduce-exp02 --methods rf_balanced rf_split_temporal fusion guided_fusion anchored_fusion --split test --seed 2026101203 --model qwen3.8-flash --max-requests 48 --identity configs/experiment-exp02.json
uv run gnss-research run --run runs/reproduce-exp02 --env-file .env
```

上面是同条件复现入口，不是新的未见测试。新方案或新数据需在新配置中写实际数据种子和实验标识。
锚定融合必须与对应guided视觉方法一起登记，共用响应而不新增请求；两项信息消融登记anchored_no_local/guided_no_local、anchored_no_scores/guided_no_scores。
参数调整、窗口和训练先验在协议/训练记录单独披露，源码适配不冒称完整论文复现。

## EXP01归档

日尺度N/E/U主要异常与持续形变的可审计实验。简单白噪声仿真、站点自身正常背景、单轮视觉数值融合，无统一毫米/速率预警阈值。覆盖尖峰、阶跃、暂态、持续形变与混合事件；创新假设与现场有效性仍需检验。

本轮实验：**GNSS-VLM-EXP01（协议v4）——基于站点正常背景的GNSS视觉数值融合异常识别仿真对照与消融实验**。统一名称与批次对应关系保存在[实验标识](configs/experiment.json)。名称在实验完成后补充，未改写冻结登记。

当前结果：24个独立仿真测试站，模型`qwen3.8-flash`。主实验/复测的持续形变日micro-F1分别为：默认RF **0.5321/0.5321**、纯视觉 **0.3673/0.4129**、融合 **0.3900/0.3326**。主实验平衡RF为**0.5789**，其`class_weight=balanced`调整单独标注。**尚未证实融合的稳定优势**；不是现场验证。

- [实验协议、对照与消融](docs/experiment-protocol.md)
- [核实论文、原始源码及适配](docs/literature.md)
- [位移优先与缺测](docs/landslide-design.md)
- [状态和真实运行结果](docs/project-status.md)
- [结果表与复测](results/main-v4/results.md)
- [研究判断与失败分析](docs/research-assessment.md)

## 运行

Python 3.11+、uv、Git。依赖与上游提交分别锁定。模型配置放本地`.env`，字段见`.env.example`。

```powershell
uv sync
uv run gnss-research setup-sources
uv run gnss-research generate --out data/new-pilot --seed 2026101201 --per-split 6
uv run gnss-research register --dataset data/new-pilot --out runs/new-pilot --methods random_forest rf_balanced extra_trees isolation_forest visual_only fusion fusion_no_reference fusion_n_only --model qwen3.8-flash --max-requests 24
uv run gnss-research run --run runs/new-pilot --env-file .env
uv run gnss-research register --dataset data/new-pilot --out runs/new-paper-pilot --methods visualtime_gnss --model qwen3.8-flash --max-requests 36
uv run gnss-research run --run runs/new-paper-pilot --env-file .env
```

小批检查后按协议的新种子运行主测试。目录必须是新目录，已冻结运行禁止覆盖或隐式重试。输出包含冻结登记、训练参数、请求/响应/模型与token、同掩码图和CSV、逐站结果、分类型/事件指标、场景分层、成对区间。当前视觉契约不预测内部阶段。失败与未执行显式计数；不复现未公开的代理代码，不伪造结果。

```powershell
uv run gnss-research evaluate --run runs/main-v4 --dataset data/main --out runs/main-v4/evaluation-new-replay
uv run python scripts/replay_author_parser.py --run runs/main-v4 --dataset data/main --out runs/main-v4/author-parser-new-analysis
uv run python scripts/audit_artifacts.py --run runs/main-v4 --repeat runs/main-repeat-v4 --upstream-repeat runs/main-upstream-repeat-v4
uv run python scripts/report_results.py --run runs/main-v4 --repeat runs/main-repeat-v4 --author-analysis runs/main-v4/author-parser-analysis --author-repeat-analysis runs/main-upstream-repeat-v4/author-parser-analysis --evaluation evaluation-corrected --out artifacts/new-report --supplement
uv run pytest -q
uv run ruff check src tests scripts
```

分支从提交`81a4016`重整。旧网页、探索脚本和结果稿不属于当前入口，原工作区和旧分支保留。当前入口没有旧API或数据兼容层。

论文基线分为原提示/源码、GNSS输入适配、作者原解析和本项目严格解析；使用保存响应的作者解析重放单独标注，不覆盖严格解析。VLM4TS已核查并锁定源码，但完整模型尚未运行，结果表不填成绩。当前结果仅支持合成数据上的研究判断。

补充包包含全部开发批次、失败、原始请求/响应、数据、冻结源码与论文证据。换机器后用`evaluate --dataset`或作者解析脚本的`--dataset`显式指定解压后的数据目录，仍校验原登记哈希，不修改注册文件。全新复现实验需创建新数据/运行目录，运行本地`.env`配置的模型；已用测试集只支持响应重放，不能作为改进方法的未见测试集。
