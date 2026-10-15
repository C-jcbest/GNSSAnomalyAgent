# EXP17 文献实读、假设与适配边界

2026-10-15，ARS academic-research-suite实验工作流；研究问题是输入表示与读图流程能否
带来真正的GNSS视觉增量。不是作者原法的完整复现实验。

| 来源与实际读取位置 | 已确认事实 | 本轮适配及不能推出的结论 |
|---|---|---|
| TAMA，arXiv:2411.02465v1，既有全文§3多尺度复核、Appendix A.3逐项重读 | 作者用新局部图复核；建议清晰刻度与辅助网格，图像不超过2000×768是其GPT-4o使用建议 | 测试相对日局部图、每日网格；不把原模型尺寸限制套用Qwen，不复制置信阈值/point-adjusted F1 |
| VisualTimeAnomaly，arXiv:2502.17812v1，§3多变量表示与§4维数影响重读 | 逐变量子图、信息密度及子图分辨率会影响表现；变量异常任务可省略坐标，不代表定位任务也可省略 | 多候选单图与拆图比较；GNSS配对不是该论文的变量异常任务，保留坐标 |
| DePlot: One-shot visual language reasoning by plot-to-table translation，arXiv:2212.10505，2026-10-15取得HTML；摘要/§1/§3/§4 | 分离图表转线性表格与LLM表上推理，使用专门训练的转换模型 | staged_readout借分工思想，同一Qwen读数再文字判断；未加载DePlot权重、未复现其ChartQA结果，非近期时序论文 |
| ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild，arXiv:2407.04172v1，2026-10-15取得HTML；摘要/§1及图像/表格例子 | 作者指出只依赖表格生成指令会丢失视觉趋势，方法从图像生成指令数据 | 因而除条形统计仍测试实际局部曲线；未训练/执行ChartGemma，不由其图表成绩预言GNSS收益 |
| AnomLLM，ICLR2025，既有全文§4.2与结论 | 图像、模型与提示影响需控制，显式CoT没有稳定收益 | 两阶段只记录可核查读图事实，不把长思维链当创新或默认有效 |
| VLM4TS，AAAI2026，DOI10.1609/aaai.v40i26.39319，既有作者源码/方法读取记录 | ViT4TS筛选与VLM核验分工 | 沿用数值候选+局部视觉核验；本项目PELT/RF不冒称ViT4TS原法 |
| Qwen官方vision文档，2026-10-15重新取得高分辨率/视觉token段及OpenAI兼容请求示例 | 文档写32×32像素/token、默认max_pixels=2621440；image项可传max_pixels；上限可增 | 本轮请求8388608，记录实际usage；首站bar_low image_tokens=930，bar_high=3772。确认了可见用量增加，不能验证代理底层权重身份 |

新增全文和检索原响应见`papers/exp17-visual-resolution/`；retrieval.json包含URL、状态与SHA。
Crossref“multimodal visual time series anomaly detection 2026”结果只作元数据筛查；
检索返回的其他题名没有阅读全文，不据标题填补方法、成绩或声称已复现。

分辨率/提示对照：bar_high与bar_grid相同3772图像tokens，axis_prompt使用与bar_grid
完全相同的图，只增逐轴判别指令。pair_tiles还改变逐图自动纵轴范围，虽然每图有
数字刻度且单图内共轴，但跨图片并非完全同范围；不能把差异只归因布局，
提示沿用了“all panels share the same vertical scale”，这一跨图表述不够准确，结果保留。
这是后续需修正的实际表示问题，不覆盖已登记请求或据其结果声称单因素因果。

首站staged_readout的18个三分量读数均与图中两位小数标签一致，但其短观察在P1/P2
误述符号，第二阶段没有选择P3。可见读数正确不等于配对判断正确。
局部图有响应照抄提示中的短evidence占位语；结构解析成功不代表观察依据充分。
本轮不会将这些响应当成可靠物理解释写进论文。

## EXP18/19补充

EXP18采用局部前后观测的散点/箱线及一个训练站的匹配/不匹配示例。后者借TAMA
参考学习思路，但把正常参考改成带异常标签对照，是必须披露的新适配。训练示例用
机制真值确定边缘，目标仍用数值估计边缘，存在训练/推断差异。箱线图压缩了时间
顺序；JSON包含同一组原始样本及索引，不将它称严格等价的时间可视化编码。

新增取得Qwen3-VL Technical Report(arXiv:2511.21631v1)全文，实读摘要、§1/§2
的动态分辨率/融合架构及§4.5图像思考部分；官方API文档混合思考段确认
qwen3-vl-plus默认false、支持enable_thinking和thinking_budget。技术报告讨论的是
发布模型家族，不足以独立核验当前代理服务的底层模型权重。
EXP19固定输入，仅请求enable_thinking=true、thinking_budget=2048；实际12个响应
均含可见reasoning_content，合计23426个reasoning tokens。该信息用于确认模式
确有不同响应，不把更长过程作为更可靠判断的证据，也不在正文转录长推理。
旧48站四类日F1由.7866降为.7840；同一指标未改善，不把事件F1小升改为成功标准。

本轮全部方案是GNSS输入/核验适配。参考论文源码/原方法成绩仍见既有EXP06档案；
未新增执行DePlot、ChartGemma、TAMA、VLM4TS的完整作者方法，不增加虚构对比行。
