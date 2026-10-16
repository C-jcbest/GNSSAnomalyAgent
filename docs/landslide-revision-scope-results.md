# 冲突范围限定修订：机制探索与新冻结确认

2026-10-09。此轮只改变局部回答的修订权限，保持初判、选窗、局部图、提示、回答与调用预算；不接入上一轮失败的整段过滤。已有三批受控记录均已参与机制选择，缓存重放只作为开发证据。另生成case0073–0084，用生成前冻结的候选和判定条件确认；受控潜在参考仍不是独立专家观测参考。

## 改动与最少消融

原方法用局部回答改写整个选定180日目标。新候选先计算与选窗完全相同的逐日原始运动冲突集合，将冲突日扩展±49日，并与有观测日和选定目标相交；只有该集合内的日期能接受局部改写，其余保留全局初判。49日来自最大42日中心间隔加7日半块的原始观测覆盖，并非扫描测试分数得到的阈值。

数值仅决定可修订范围，不直接将日期判为活动或静稳。失败／未知局部回答在有修订权限的日期保持未知；缺测永远未知。以最终活动重新门控共同学习阶段头，不在范围外复制阶段内容，也不把弱运动强制赋S。

关键对照为整目标改写、冲突范围改写，以及每目标与冲突范围具有**相同可修订有观测日数**的居中改写。居中对照按距目标中心排序，等距选较早日期。它检验收益来自冲突位置还是仅仅改动日期减少。三种范围分别应用于固定纯图、自适应纯图和自适应数值文字；直接视觉及六项数值基线完整保留。新确认另加入同输入、共同训练正常参考的TAMA GNSS适配初判和阳性复核，两遍均报告。

候选函数和业务测试分别位于 `scripts/probe_landslide_revision_scope.py` 和 `tests/test_landslide_revision_scope.py`。冻结主runner和原默认未修改；范围脚本是可复现的确定性候选执行器，直接接受已冻结原始流程输出，不新增模型请求。

## 三批已见缓存：六遍默认的开发证据

产物为 `artifacts/landslide-adaptive-review-2026-10-09/revision-scope-probe-v1/`。没有新增API调用，也没有重复随机回答后挑选。

| 已见数据／遍次 | 整目标F1 | 冲突范围F1 | 撤回FP日 | 损失TP日 |
|---|---:|---:|---:|---:|
| 方法选择，1 | .855411 | .855596 | 1 | 0 |
| 方法选择，2 | .852623 | .861772 | 51 | 0 |
| 原默认确认，1 | .722700 | .728000 | 33 | 0 |
| 原默认确认，2 | .842642 | .842642 | 0 | 0 |
| 整段过滤追加确认，1 | .785182 | .796314 | 60 | 0 |
| 整段过滤追加确认，2 | .836470 | .836959 | 1 | 0 |

六遍默认弱召回均未下降，最后一遍另恢复1 TP。居中等预算对照有时损失TP或增加FP，但原默认确认第二遍居中F1 .846842高于冲突范围；开发结果不支持全面优势。所有流程、遍次及现场配对表示共390个目标比较，274个范围缩小，可修订有观测日48119／整目标67768。该总数包含同回答不同流程和现场配对表示，不能当390独立样本或解释为节省API。

无标签现场重放没有解决主要季节响应，不计算撤回FP或准确率。此前严格首／中／尾支撑过滤的否定结论继续保留，不降低门槛、不并入新候选。

## 新确认的生成前冻结

`revision-scope-confirmation-data-v1/protocol.json` 在生成case0073–0084前保存候选源码SHA、固定策略、来源绑定和停止条件。仍为六族×2记录、730日；共同train24／validation12的规则参数和学习权重不变。四主视觉流程同Qwen、参数、画法、180日目标、最多两次复核，每流程两遍。TAMA仅使用原共同train的0001／0007／0013正常参考，先冻结所有主／TAMA回答，再读参考评分。

预先条件为：任一遍损失TP、弱召回下降、FP增加或F1下降均不升级；**两遍必须各至少净减少1 FP**，否则拒绝或保留未决。不能用新记录重新选择范围宽度、提示、重复或数值阈值。等预算居中消融完整报告，不能在确认后改成新的主候选。

统一汇总代码 `scripts/analyze_landslide_revision_scope_confirmation.py` 核查生成前候选源码、数据manifest、训练绑定、主／TAMA／范围预测冻结、相同输入与共同阶段头；独立sklearn重算活动与阶段指标，同时核查事件匹配与边界一致性。报告逐日P/R/F1、弱召回、逐日TP保护及FP变化、事件IoU .3/.5、匹配条件边界、未知覆盖、端到端及真活动内阶段。模型成本按真实唯一回执统计，共享回答不跨方法重复相加。范围成本单独测原始证据＋掩码＋合并，排除VLM、绘图和共同阶段头，不冒充完整数值CPU基准。

## 复现命令

以下输出目录均已占用后不能重复生成或重跑API；只读分析应另命名。实验环境与凭据读取复用原runner，`.env`不导出。

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
$py='artifacts/reproduction-2026-10-07/environment/Scripts/python.exe'
$base='artifacts/landslide-adaptive-review-2026-10-09'
& $py scripts/prepare_landslide_revision_scope_confirmation.py --out "$base/revision-scope-confirmation-data-v1"
& $py scripts/run_landslide_adaptive_review.py prepare --config configs/landslide-revision-scope-confirmation-v1.json --out "$base/revision-scope-confirmation-v1"
& $py scripts/run_landslide_tama_adapter.py prepare --source "$base/revision-scope-confirmation-v1" --out "$base/tama-gnss-scope-confirmation-v1"
& $py scripts/run_landslide_adaptive_review.py run --out "$base/revision-scope-confirmation-v1"
& $py scripts/run_landslide_tama_adapter.py run --out "$base/tama-gnss-scope-confirmation-v1"
# 全部回答冻结后：
& $py scripts/run_landslide_adaptive_review.py score --out "$base/revision-scope-confirmation-v1"
& $py scripts/probe_landslide_revision_scope.py --run "$base/revision-scope-confirmation-v1" --additional-baselines "$base/revision-scope-confirmation-data-v1/additional-baselines/predictions.json" --out "$base/revision-scope-confirmation-v1/scope-v1"
& $py scripts/analyze_landslide_revision_scope_confirmation.py --run "$base/revision-scope-confirmation-v1" --tama "$base/tama-gnss-scope-confirmation-v1" --scoped "$base/revision-scope-confirmation-v1/scope-v1" --out "$base/revision-scope-confirmation-v1/analysis-v1"
& $py scripts/analyze_landslide_revision_scope_cases.py --run "$base/revision-scope-confirmation-v1" --scored "$base/revision-scope-confirmation-v1/analysis-v1" --out "$base/revision-scope-confirmation-v1/case-analysis-v2"
```

## 新确认结果：通过小样本筛选，收益集中于一条记录

全部输出先冻结，30配置×12共同记录统一评分。解释图v1全记录纵轴压缩弱局部趋势，v2按实际显示范围计算纵轴后重新渲染，v1及其源码保留。两遍均通过预先条件：冲突范围各撤回18 FP，损失／新增TP均0，新增FP也0，弱召回保持不变。因此保留范围机制作为论文方法候选，**不自动改动历史冻结默认或声称已完成独立现场验证**。

| 方法 | P，1／2 | R，1／2 | F1，1／2 | 弱召回，1／2 | FP，1／2 | 调用，1／2 |
|---|---:|---:|---:|---:|---:|---:|
| 直接视觉 | .9475／.8435 | .5896／.5971 | .7269／.6992 | .2583／.2583 | 74／251 | 12／12 |
| 固定纯图、整目标 | .9469／.8558 | .6222／.5891 | .7510／.6979 | .2986／.1860 | 79／225 | 36／36 |
| 原自适应纯图、整目标 | .9495／.9479 | .6805／.6823 | .7928／.7934 | .4419／.4348 | 82／85 | 23／24 |
| **自适应纯图、冲突范围** | **.9601／.9585** | **.6805／.6823** | **.7965／.7971** | **.4419／.4348** | **64／67** | **23／24** |
| 自适应纯图、等日数居中 | .9370／.9501 | .6699／.6726 | .7813／.7876 | .4313／.4242 | 102／80 | 23／24 |
| 数值文字、冲突范围 | .9832／.9850 | .6712／.6955 | .7978／.8153 | .3863／.4988 | 26／24 | 23／24 |
| TAMA适配、正常参考初判 | .8966／.9442 | .6200／.5830 | .7331／.7209 | .3993／.2583 | 162／78 | 13／13 |
| TAMA适配、阳性复核 | .9567／.9409 | .6143／.5834 | .7482／.7202 | .3472／.2583 | 63／83 | 23／21 |

| 共同训练／验证数值基线 | P | R | F1 | 弱召回 | 事件F1，IoU .3／.5 | 端到端阶段macroF1 |
|---|---:|---:|---:|---:|---:|---:|
| robust规则 | 1.0000 | .3835 | .5544 | .0000 | .7273／.2727 | .5493 |
| PELT规则 | 1.0000 | .2193 | .3598 | .0000 | .5263／.1053 | .3401 |
| XGBoost | .9401 | .5750 | .7136 | .2453 | .6207／.3448 | .7207 |
| robust活动＋学习阶段 | 1.0000 | .3835 | .5544 | .0000 | .7273／.2727 | .5558 |
| 同证据多尺度原始规则 | .9480 | .6752 | .7887 | .3685 | .5556／.4444 | .7180 |
| 同证据长尺度XGBoost | .9536 | .6072 | .7420 | .3400 | .7143／.5000 | .7395 |

原始完整P/R/F1、范围全部消融、覆盖、真活动内阶段、事件／边界和成本在 `revision-scope-confirmation-v1/analysis-v1/summary.csv`，逐记录在`per-record.csv`。此批纯图范围F1略高于多尺度规则，但差值仅.0078／.0085、配对区间均跨零；前两批该规则更高，不能改写成稳定数值优势。数值文字两遍F1更高，第一遍弱召回却更低；不在确认后把文字组切为主候选。固定和数值文字的冲突范围在此批没有改变活动阳性，范围效果具有流程依赖。

### 真正修正的位置与失败范围

两遍各18个撤回FP均来自同一条强后弱记录case0078：局部目标`[450,630)`，整目标回答把活动提前到450日；冲突最早从519日开始，49日扩展使修订权限从470日开始。新结果在`[450,470)`保留原初判，撤回18个有观测的潜在静稳日，保留470日后的局部弱运动判断，两遍弱检出仍155日。不能把两遍同记录的修正当成两个独立事件或36个独立样本。

等日数居中范围两遍分别损失24／22 TP、增加34／10 FP（同时撤回14／15 FP），其中case0078各损失9个真实弱日。它支持冲突位置比该居中控制更合理，但不是与所有可能等预算位置策略的比较。整记录描述bootstrap相对整目标区间为`[0,.0110]`／`[0,.0111]`，包括零，反映收益集中而非稳定普遍改善。

限制同样明确：case0081弱恒速两遍完全漏检，case0084强后弱的弱尾两遍均检出0；同证据规则在后二者分别检出16／56弱日。TAMA在case0081第一遍检出138日而第二遍0；case0078则两遍弱日0。范围规则只能约束已有局部修订，不能使初判、数值证据和局部模型未发现的运动凭空恢复。第一遍case0084另有68个局部未知日被原全局阴性保留，覆盖改善；这与18个阳性误报撤回是不同变化，不能混为新增活动恢复。

实际`selections.json`与原始证据重算区分了两类失败：case0081两遍所有候选窗冲突均0，长证据有效535日但运动阳性0，因此没有局部请求；case0084两遍有22个弱尾漏检冲突，确实选择[450,630)，局部输出仍未确认弱运动。前者是触发依据失灵，后者是取证后解释失灵，不能笼统归为“范围过小”。这两条已见记录可用于下一项机制探索，不能再用来确认改进后的独立效果。

### 事件、阶段和成本

冲突范围在IoU .3与.5下的事件F1均为.8000／.7692（两遍），与整目标完全相同，每遍匹配10事件；匹配条件起点MAE由32.3／31.4降至30.3／29.4日，终点仍20.5／19.4日。TAMA阳性复核事件F1 .8333／.7500，直接视觉 .8182／.7826；活动F1更高不等于每个事件指标更好。

纯图范围端到端阶段macroF1 .6890／.7074，比整目标仅小幅提高；真活动内macroF1仍.6962／.7130，**完全未变**。该增益来自撤回活动外误报阶段，不支持A/S/D内容得到修复。范围覆盖.9870／.9994，原整目标.9788／.9994，未过滤参考未知日追求更高分。

纯图范围调用23／24、token74552／78711，与整目标完全相同，相对固定调用减少36.1%／33.3%。TAMA阳性复核23／21调用、237436／212702 token；不能宣称每组调用都更少。范围确定性原始证据＋掩码＋合并在本机对每遍12条一次测量共.326／.333秒，排除VLM、绘图和共同阶段头，不是隔离数值CPU比较基准。

唯一真实回执为主流程112请求／438995 token，TAMA44／450138，合计**156请求／889133 token**，无登记传输／解析失败。主线累计**782请求／3751303 token**。82份数据／范围manifest文件、12共同输入及阶段头、30独立sklearn指标与事件边界组、156实际payload／图片绑定均通过；生成前候选源hash仍相同。4项业务测试、Ruff通过，14条既有Matplotlib／Pyparsing弃用警告。汇总图、case0078实际发送局部图和案例图均已视觉检查。全部记录对照和所选成功／失败图在`case-analysis-v2`，不是独立新增证据。

新确认图：`revision-scope-confirmation-v1/analysis-v1/scope-confirmation.png`；解释图：`case-analysis-v2/case_0078-scope.png`与`case_0084-scope.png`。候选通过限定条件但增益小且集中，保留为可检验方法贡献，不再追加同批范围调参。后续优先检验原始冲突证据失灵的弱恒速／弱尾情景，候选改变必须另冻结未参与选择数据；仍不新增复杂Agent框架。

## 最近邻与主张边界

本轮实际阅读AnomaMind官方Locator／Actor／Detector／workflow／finalize源码，见[源码定位笔记](landslide-nearest-method-code-review.md)。它已有视觉粗定位、数值取证和最终修订，不能把这些通用机制当创新。所读commit未见本文的逐日冲突半覆盖修订掩码，但该观察不证明全领域首创。AnomaMind包含SFT／RL检测器，当前没有运行它，不能以临时共同Qwen替身冒称原方法。

正式观测结论仍需独立可信参考。用户已明确暂时没有合适人员、先继续自主研究；本轮继续方法开发和新受控确认，没有生成伪专家标签或再次索要复核人员。

后续[84日尺度开发检验](landslide-longer-evidence-results.md)未改善视觉识别；同活动支撑重新拟合的XGBoost42／84为.8100／.8477，高于范围候选。后续是已见开发比较，原30组冻结结果保留；不能仅据旧六数值基线宣称本批数值优势。完整稿更新v5。
