# GNSS 视觉大模型异常识别研究

已完成[29页中文Word论文初稿](manuscript/GNSS多类型异常识别论文初稿.docx)，
题为《融合数值形态候选与视觉大模型的GNSS多类型异常识别方法》。
七章正文、逐章文献参考、结果审阅与版面检查见[论文材料](manuscript/README.md)。

当前完成 **GNSS-VLM-EXP04-v1：GNSS形态分工视觉数值融合异常识别实验**。
分支`codex/gnss-vlm-research`；主方法为`morphology_fusion`。
尖峰交给数值分类，阶跃交给排名候选视觉确认，持续活动按原始位移复核边界。
先新验证小批，后24验证站选定四类等权日F1，再冻结两个新种子、48测试站。

| 方法 | 四类等权日F1 | 四类等权事件F1≥.5 |
|---|---:|---:|
| 平衡RF | .4257 | .2276 |
| 数值时间解码 | .3734 | .3542 |
| 纯视觉 | .1907 | .1904 |
| EXP03排名融合（相同新测试） | .5080 | .5045 |
| **EXP04形态分工融合** | **.5681** | **.5646** |

主要新增收益是尖峰误报减少：尖峰F1由.3913提升至.6316，召回有所下降。
固定数值活动边界变体的等权日F1=.5816、通用日F1=.7734；
主批去局部图也优于完整方法，因此论文重点放在有证据的形态分工，不强求每个模块领先。
不同变体、不同站数和不同F1分别标注，完整主方法通用日F1为.7095。

- [EXP04两个新种子完整结果](results/exp04-two-seeds/results.md)
- [主批24站对照与消融](results/exp04-main-v1/results.md)
- [EI论文使用建议与实际结论](docs/exp04-assessment.md)
- [冻结方案](docs/experiment-exp04.md)、[开发与指标选择](docs/exp04-development.md)
- [固定场景原始位移与标签图](results/exp04-cases/cases.png)

开发36+正式144，共180请求；全部失败保留，无自动重试，主方法0/48失败。
29个业务测试、Ruff及四批冻结材料核验通过。仅离线模拟GNSS证据。
原始数据/请求/响应/源码快照在`data/exp04-*`及`runs/exp04-*`，凭据不入库。
文献持续实读；作者源码、通用库、GNSS适配与未执行模型均明确区分。

## EXP04复现入口

```powershell
uv sync --locked
uv run python scripts/prepare_exp04.py --seed 2026101302 --out data/reproduce-exp04
uv run gnss-research register --dataset data/reproduce-exp04 --out runs/reproduce-exp04 --methods random_forest rf_balanced rf_split_temporal visual_only guided_fusion ranked_boundary_fusion morphology_fusion morphology_fixed_activity --split test --seed 2026101301 --model qwen3.8-flash --max-requests 48 --identity configs/experiment-exp04.json
uv run gnss-research run --run runs/reproduce-exp04 --env-file .env
```

上述复现种子已使用；新优化另设开发/测试种子，不覆盖原运行。

## EXP03历史结果

已完成 **GNSS-VLM-EXP03-v2：GNSS排序候选视觉确认、可修正活动边界与训练规模对照实验**。
分支`codex/gnss-vlm-research`；主方法结果对应`ranked_boundary_fusion`行。
两个新种子、48个独立模拟测试站，主方法形变日/事件F1 **0.7474/0.8000**，
相同数值法`rf_split_temporal`为 **0.5216/0.5854**；阶跃日F1 **0.6207**，数值法 **0.2000**。
停止后假活动日553→81，活动漏日增加；尖峰F1 **0.4561**低于数值 **0.6667**，保留误报代价。
相对数值法形变日F1差值95%CI [.0890,.3632]；相对旧锚定/自由视觉区间跨0，不宣称稳定最优。
24/96训练同站对照没有证明数值法随训练增加而改善；局部图消融仅主种子，增量未确定。

- [EXP03两个新种子结果与训练规模对照](results/exp03-two-seeds/results.md)
- [主种子消融与作者原码/适配结果](results/exp03-main-v2/results.md)
- [冻结方案和方法对应](docs/experiment-exp03.md)
- [开发取舍与负结果](docs/exp03-development.md)
- [研究结论与优先问题](docs/exp03-assessment.md)

先18次新验证小批，再冻结正式测试；确认376次模型请求，另至多1次中断在途请求状态未知。
主批在新目录仅离线恢复，未重试或覆盖；同响应派生不重复计数。不是现场验证。
27个业务测试及Ruff通过。论文与作者代码持续实读，原码、GNSS适配和未执行项分别披露。
补充包：`artifacts/exp03-package/exp03-supplement.zip`；完整数据、冻结源码、请求/响应、
中断原批与恢复、开发负结果和文献证据均保留。清单与SHA见
[补充包记录](results/exp03-two-seeds/supplement-manifest.json)，本地凭据及Git元数据排除。

## EXP03运行

```powershell
uv sync --locked
uv run python scripts/prepare_exp03.py --seed 2026101207 --out data/reproduce-exp03
uv run gnss-research register --dataset data/reproduce-exp03 --out runs/reproduce-exp03 --methods rf_split_temporal guided_fusion anchored_fusion boundary_fusion ranked_boundary_fusion --split test --seed 2026101206 --model qwen3.8-flash --max-requests 24 --identity configs/experiment-exp03.json
uv run gnss-research run --run runs/reproduce-exp03 --env-file .env
```

上述是同条件复现入口，不是未见测试。额外训练站不改变验证/测试观测；
生成的`-train24`数据集是前24训练站的显式子集，`-small`是首六验证站的小批。
排名点确认不要求RF票决阳性，但必须是已有排名峰且视觉同日同类型确认，不作日期吸附。
所有窗口/训练先验/排名预算/数值默认判决均披露，未设跨站点统一物理告警阈值。

## EXP02历史结果

已完成 **GNSS-VLM-EXP02-v3：GNSS活动与质量异常分离的时间解码及局部视觉复核实验**。
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
