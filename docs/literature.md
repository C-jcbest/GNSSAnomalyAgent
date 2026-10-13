# GNSS 视觉异常识别：文献与官方代码核查

## EXP02优化过程补查（2026-10-12）

重新核对VLM4TS全文引言和附录Tables 7–9讨论：长全局图存在分辨率/上下文矛盾，
候选核验通常提升精度但可能降低召回，部分数据可增补漏候选；其原文是其他领域实验，
不能推定本轮GNSS同样改善。作者源码 `verify_intervals` 确实查看全图及候选并可修改区间。
重新核对TAMA的Multiscale Self-reflection及正常参考消融：其局部放大/复核已有先例，
主表包含point-adjusted指标，而部分消融另报无PA结果；本轮继续使用无PA评分，不能跨协议引用论文数值作直接比较。

本轮实测自由视觉复核可能误报普通噪声及混淆暂态/阶跃，采用局部原始图与正常历史拟合参考是待验证改进。
正常参考、两阶段复核和数值工具结合都不是本项目首次提出。
另外核实sklearn官方[RF文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
与[MultiOutputClassifier文档](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)：
多输出类别权重逐列相乘；独立二元估计器是另一训练方式，不能描述成仅改接口。
对应页面已缓存到证据清单，当前安装源码也确认此权重语义。独立训练在新验证集上反而退化，保留负结果，未宣称必然改善。
本项目Viterbi路径使用RF后验作启发式证据，不是另取得或复现了某篇GNSS HMM论文。

核查日期：2026-10-12。本文由 AI 辅助查阅，事实依据限定为实际取得的论文、DOI 元数据和官方代码；没有执行上述论文的完整复现实验。本文不是系统综述，也不构成创新性证明。所有来源字节、URL、访问日期、SHA-256 与抓取失败记录见 `papers/evidence/manifest.json`。`metadata`、`metadata_abstract`、`full_text`、`code_documentation`、`code` 和全文提取文本分开记录。

## 1. Can Multimodal LLMs Perform Time Series Anomaly Detection?

- 作者：Xiongxiao Xu、Haoran Wang、Yueqing Liang、Philip S. Yu、Yue Zhao、Kai Shu。
- 论文：[arXiv 2502.17812 v2 全文](https://arxiv.org/html/2502.17812v2)；[ACM DOI 10.1145/3774904.3792376](https://doi.org/10.1145/3774904.3792376)。Crossref 元数据载为 *Proceedings of the ACM Web Conference 2026*，出版日期 2026-04-12。
- 官方代码：[mllm-ts/VisualTimeAnomaly](https://github.com/mllm-ts/VisualTimeAnomaly)，由论文 v2 正文脚注明确链接。核查提交 `7158c4ff05bc6fb27bf5045a1e8129318ca4a47d`；本地只读上游位于 `vendor/upstream/visualtime/`。
- **来源冲突**：上述提交的 README 写 `WWW'25`，引用项也是 2025 预印本；这不能替代论文 v2 与 DOI 的 2026 正式发表信息。引用时明确所用版本。

### 已核实实验事实

v2 §2.2 使用正弦/余弦合成序列注入异常，同时在 UCR Symbol / 官方代码命名 `Symbols` 和 UEA ArticularyWordRecognition 上注入异常作为案例；这些不是 GNSS 滑坡数据。单变量长度 400、多变量每变量长度 200；每个异常类型生成 100 张图，实验重复 3 次。§3 将点和区间异常比例设为 5%，区间长度 20；变量异常是在 20 个变量中注入一个异常变量。

v2 §4 只在 global、trend、random 三类上考察随机删点率 0%、5%、10%、15%、20%、25%。传统方法比较采用均值插补，视觉输入保留空白。随机散点缺失的发现不能直接推广为连续 GNSS 断链、长期缺测或高度不规则采样的验证结果。

v2 §3 和 §6 明确指出视觉模型对精确点异常定位较弱，并提出 TSAD-Agents：Scanning、Planning、Detection、Checking 四个代理，共享动态记忆，使用 Gemini、LangChain、iForest 和视觉模型工具。因此“视觉 LLM + 数值方法 + 智能体”已经存在，不能单凭组合宣称本研究创新。

### 官方代码实际提供什么

上述提交提供 `src/generator.py`、`src/main.py`、`src/prompt.py`、模型 API、数据加载与结果聚合脚本；**未找到论文 v2 的 TSAD-Agents 四代理实现**。不能把本地另写的 LangChain 智能体标成已复现的官方 TSAD-Agents。

入口是从上游 `src/` 工作目录执行 `python main.py --category synthetic --scenario univariate --model_name gpt-4o --data global`，默认 `--variant 0shot-vision`。`config.py` 实际只注册该视觉零样本配置。`main.py` 同时导入 GPT、Gemini、LLaVA、Qwen 实现，使用 `TSIDataset` 读取 `data.pkl`（字段 `series`、`anom`，不规则单变量另有 `drop_index`），并读取固定相对目录的 PNG。

官方图像提示由 `data_name` 选择 `PROMPT_POINT`、`PROMPT` 或 `PROMPT_VARIATE`，分别输出点索引、包含两端的区间、异常变量 ID。**这包含了异常粒度的先验选择**，不是未知 GNSS 异常类型的统一分类接口。比较时必须披露这个条件，不得由测试真值替模型选择提示。

README 列出 Python 3.10.14、torch 2.4.1、numpy 1.26.4、transformers 4.49.0.dev0、huggingface-hub 0.24.7、openai 1.44.0、google-generativeai 0.8.3；源码还导入 scipy、loguru、yaml 等。没有完整锁文件，不能宣称环境已经可复现。`openai_api.py` 实际用 `AzureOpenAI` 和工作目录的 `credentials.yml`，不是直接读取标准 OpenAI 环境变量。凭证应按上游契约在忽略目录内提供，不提交密钥。

**GNSS 适配边界**：官方程序可运行其数据布局的图像实验；要用于 GNSS，需要显式转换序列和图像目录、选定提示契约、还原索引到原始时间戳及处理缺测掩码。复用原文提示与请求部分的轻量适配器应叫“VisualTime 官方提示适配版”，不能叫“完整官方代理原版”。修改 GNSS 图形、提示、模型服务、解析或指标均须进入适配记录。源码 MIT LICENSE 已核查。

## 2. Harnessing Vision-Language Models for Time Series Anomaly Detection

- 作者：Zelin He、Sarah Alnegheimish、Matthew Reimherr。
- 已读取：[arXiv 2506.06836 全文](https://arxiv.org/html/2506.06836)。官方 README 标注 `AAAI 2026 Oral`；本次未另取得正式会议 DOI，引用可使用已核实的 arXiv 版本，不能把 README 标记当作独立出版核验。
- 官方代码：[ZLHe0/VLM4TS](https://github.com/ZLHe0/VLM4TS)，核查提交 `8ab8c16414eb2c1a861dfc3e76f458180035a879`；本地原样源码 `vendor/upstream/vlm4ts/`。

### 已核实实验事实与限制

论文设计是两阶段：ViT4TS 用多尺度视觉 patch 比较产生候选，VLM4TS 用长时间视图核验候选。主要实验覆盖 NASA MSL/SMAP、Yahoo A1–A4、NAB 五组数据，共 11 个数据集；主协议以单变量为主，附录有初步多变量扩展，**没有本次 GNSS 滑坡适用性验证**。

Appendix A 的预处理是 min-max 归一化与线性去趋势；Yahoo A4 另在已知变点前后标准化。论文对分数做 EWMA，阈值为 `mean + norm.ppf(1-alpha) * std`，`alpha ∈ {0.10, 0.01, 0.001}`，报告 contextual F1 和这些设置上的 F1-max。这里的阈值基于分数分布，不是固定毫米阈值，但仍存在阈值超参数。不能把它描述成“无阈值”，也不能把测试标签选出的 F1-max 当作部署性能。

### 可以直接调用的接口

```python
from src.models.vit4ts import ViT4TS
from src.models.vlm4ts import VLM4TS

# data: pandas.DataFrame，列 timestamp、value；一次一条单变量序列。
scores, timestamps = ViT4TS().predict_scores(data)
intervals = ViT4TS().detect(data)
verified_intervals = VLM4TS().detect(data)
```

返回区间列为 `start`、`end`、`severity`；时间戳按时间排序后映射。默认参数由源码核实：`window_size=224`、`window_step_ratio=4.0`（步长 56）、`standardize=True`、`alpha=0.01`、CLIP `ViT-B-16`。默认 VLM 名称 `gpt-4o`，OpenAI 客户端读取 `OPENAI_API_KEY`；论文实际使用 `gpt-4o-2024-08-06`，未固定快照的仓库默认调用不能称为同一个模型实验。`detect` 若候选为空，会直接返回，不调用 VLM。

`requirements.txt` 固定 torch 2.9.1、torchvision 0.24.1、numpy 2.2.6、pandas 2.3.3、matplotlib 3.10.8、open_clip_torch 3.2.0、openai 2.11.0、scikit-learn 1.7.2、scipy 1.15.3、Pillow 12.0.0 等，应在独立环境安装。API 调用、模型权重下载与完整推理本次未执行。

**GNSS 适配边界**：直接输入至少 224 点的完整单分量或明确的位移派生量，原版保留默认去趋势和分数阈值。`orion_to_internal` 只排序、不插补；默认预处理使用 `scipy.signal.detrend`，不能推定支持 NaN。滑窗图按索引生成，原始不规则时间间隔不进入 ViT 图的几何结构。原版缺测实验需要预先定义统一输入协议；分段、插补、缺测映射、ENU 融合、关闭去趋势或增改提示均属于适配，分别报告。特别是去趋势对持续形变的影响应以实验检验，不能先假定有益。

上游 README 声称 MIT License，但此提交代码树没有独立 LICENSE 文件；已保留来源记录。当前只读运行可继续评估，正式重新分发时应保留声明并进一步核查授权文件。

## 3. GNSS discontinuity：领域证据而非可直接复现的官方基线

Laura Crocetti、Matthias Schartner、Benedikt Soja (2021). *Discontinuity Detection in GNSS Station Coordinate Time Series Using Machine Learning*. *Remote Sensing*, 13(19), 3906. [DOI](https://doi.org/10.3390/rs13193906)。题名、作者与时间从 Crossref 核实，23 页官方 PDF 从 [MDPI 内容服务器](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-13-03906/article_deploy/remotesensing-13-03906.pdf) 取得。

研究是日本 GNSS 地震阶跃的监督分类，不是滑坡持续形变识别。§3 按站点 80%/20% 拆分（532 训练站、133 测试站），采用三折交叉验证；§4 比较十种机器学习算法，21 天滑窗、三分量归一化加幅差特征对其 Random Forest 方案有帮助。可借鉴独立站点拆分和 ENU 相对形态，但不能把这些设置直接当作 GNSS 滑坡的最优参数。

**必须披露的相反证据**：原文 §3 对位移变化至少 10 mm 的阶跃标真，§4.4 扫描 5–15 mm 并选择 10 mm。该文具有固定的位移标注阈值，不能用于支持本项目“所有站点不共用物理阈值”的规则。应沿用原文论证边界：阶跃检测与地震位移筛选。若另写归一化 Random Forest 方案，应叫“受该文启发的 GNSS 分类器”，不能写“官方源码复现”。

全文 Data Availability 只给 Nevada Geodetic Laboratory 数据与断点目录 [geodesy.unr.edu](http://geodesy.unr.edu/)，本次没有找到论文明确声明的公开源代码仓库。这是检索边界内未找到，不是断言不存在。摘要写 525/565 地震正确分类，正文 §4.3 写 524/565，本文记录原文冲突，不将任一数值当作本项目结果。

## 4. 可追溯的数值/机器学习对照候选

| 候选 | 核实来源与官方实现 | 在本研究中的边界 |
|---|---|---|
| 离线变点检测 `ruptures` | Truong、Oudre、Vayatis (2020), *Selective review of offline change point detection methods*, *Signal Processing*, 167, 107299；[DOI](https://doi.org/10.1016/j.sigpro.2019.107299) 与[官方仓库](https://github.com/deepcharles/ruptures) README 互相对应；查询 HEAD `ee1c8ff8a548d54c641b2bb471562165931f31c7` | 作者维护的通用变点库，不是天然完整 GNSS 异常分类器。PELT 的 `pen` 与核/成本选择必须记录；README 示例 `pen=10` 不是论文给 GNSS 的推荐值。只输出断点时不可伪称已定位完整异常活动区间。 |
| Anomaly Transformer | [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer)，README 关联[ICLR 2022 OpenReview](https://openreview.net/forum?id=LzQQ89U1qm_)；查询 HEAD `b0ee470c8012bff857fb600462aec6209c4a18d9` | 本次只核查作者仓库 README，未读论文全文，也未执行模型。官方评估明确使用 point adjustment，若后续加入，必须同时保留原论文指标和无调整主指标；不能以其经真值调整的预测作为 GNSS 部署性能。 |

未安装或运行上述方法的地方不填成绩。对于 sklearn IsolationForest / OneClassSVM，应明确“通用库实现”，除非另外取得作者源码并核查；引用一篇论文不自动使现有库代码成为论文原代码。

## 5. 本分支可检验的研究方向与持续核查

以下是**研究假设与建议**，不是已验证文献结论：围绕 GNSS 持续活动区间，将原始位移的相对变化、站点内尺度信息和数值候选提供给视觉模型，考察能否减少尖峰/仪器阶跃被解释为持续形变、以及导数提前响应造成的过早标签。活动区间与加速/减速子阶段分开评价；缺测位置不赋确定形变标签。是否优于原版视觉模型和数值方法，必须由独立种子、独立站点或独立场景的对比与消融证据决定。

创新候选应聚焦 **GNSS 业务区分与跨站点相对判据的可检验收益**；“用 LangChain”“加一个数值工具”“生成曲线图”不构成独立新颖性依据。本次只读到有限文献，未作穷尽检索，不能声称首次。

持续核查纳入每个阶段：选定基线后更新官方提交、模型快照与源码契约；实现适配时记录每个数据/参数/提示/指标差异；查看异常个例时回查相关文献是否已研究该失败模式；论文写作前再核查新发表 GNSS/VLM 时序文献。新的事实只在取得实际来源后进入此表，失败抓取仍保留限制记录。

主实验禁止按测试标签选择阈值或注入位置提示，固定超参数如上游默认 `alpha` 须明示；合法的站点内分数标准化与统一毫米告警阈值不是同一概念。若报告 oracle F1-max，放独立的上界诊断栏并注明测试标签参与选择。原版默认、GNSS 数据适配、方法改动和新分支运行结果分别标识，旧分支结果不迁作本分支成绩。

## 6. 实现后补充：正常参考、GNSS 运动与质量标签的最近邻

补充日期同为 2026-10-12。本轮为有界的近邻核查：沿 VisualTimeAnomaly 引用读取 TAMA，沿 Crocetti 引用核查 GNSS 趋势/断点分解工作，并以 Crossref 查询 `GNSS time series anomaly detection outlier offset`，限制 2023-01-01 至 2026-10-12、只取相关性排序前 8 条，选择与多站点 GNSS 异常检测最接近的 TimesNet 工作进一步读取。最终深入核查的是以下 **3 项工作**，不是数据库穷尽搜索；Crossref 相关性排序不是发表时间排序，查不到不表示不存在。

### 6.1 TAMA：正常参考与语义异常分类已经有直接先例

已读取 [Zhuang 等的 arXiv:2411.02465v1 全文](https://arxiv.org/html/2411.02465)，2024-11-04，题名 *See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers*。取得的全文是 v1；页面残留 `DOI: XXXXXXX.XXXXXXX` 和模板会议文本，不是有效 DOI 或出版信息。不得据此声称正式会议发表。

§3.2 的三步流程为 Multimodal Reference Learning、Multimodal Analyzing、Multi-scaled Self-reflection：用正常片段图像提取正常模式描述，再利用描述检测新窗口的异常区间、类型、置信度与解释，最后利用局部放大图纠错。正文使用 point、shapelet、seasonal、trend 四类；附录提示进一步列 global、contextual、frequency、trend、shapelet。正常参考提示明确要求来自同一数据通道、不同位置的片段，描述 period、stability、trend、peak、trough。窗口重叠输出经过置信度累加、类型投票与置信阈值汇总。§5.3 还有正常/异常参考对照，§6 明确本研究以单变量为限。

**已确认重叠**：同一通道正常参考、视觉时序分析、语义异常分类、区间解释与正常参考消融都已有先例，不能算作本分支单独首次提出的概念。TAMA 也有不启用 self-reflection 的 `TAMA*` 变体，仅删去复核步骤不能自动形成创新。

**仍未确认**：已读取部分没有建立本分支的“GNSS 持续运动活动区间与尖峰/阶跃/暂态质量形态可以同时存在”的两轴标签契约，也未确认使用站点正常参考数值统计与视觉证据做同一个 GNSS 单轮融合。这里描述的是已读取证据边界，不是证明 TAMA 或其后续工作不存在这些能力。应比较正常参考如何选取、正常位移变化是否被错误抹除、类型是否互斥和有无真值先验，而不只比较代理数量。

### 6.2 GNSS 分解工作：趋势、断点与异常值分开并非全新领域思想

Heflin 等 (2020)，*Automated Estimation and Tools to Extract Positions, Velocities, Breaks, and Seasonal Terms From Daily GNSS Measurements: Illuminating Nonlinear Salton Trough Deformation*，[DOI 10.1029/2019EA000644](https://doi.org/10.1029/2019EA000644)。本次核实 Crossref 元数据与摘要；Wiley 全文页面被验证挑战阻断，**没有读取此文全文**。

摘要明确自动处理断点检测和异常值剔除，拟合 positions、velocities、breaks、seasonal terms，并分离 interseismic、annual、coseismic、postseismic 信号；描述 GeoGateway 中的可视化分析工具。它足以反驳“GNSS 运动趋势与断点/异常值分别处理这一思路从未存在”的宽泛论断。摘要不能支持更细的多标签定义、候选合并细节、站点正常参考协议或视觉 LLM 单轮融合；这些保持未确认，也不能仅凭摘要未提及 LLM 就作完整方法排除。

### 6.3 2025 GNSS TimesNet：真实多站点数值学习近邻

Andrade、Paulo、Marques 的 *Anomaly detection in weekly SIRGAS GNSS time series: a TimesNet assessment in real data*：正式发表已由 Crossref 核实为 *GPS Solutions*，2025-03-15，[DOI 10.1007/s10291-025-01839-3](https://doi.org/10.1007/s10291-025-01839-3)。本轮实际读取的是 2024-11-15 [Research Square v1 全文](https://www.researchsquare.com/article/rs-4460921/v1)，[预印本 DOI](https://doi.org/10.21203/rs.3.rs-4460921/v1)；该 PDF 标明正式版本关系，但**正式版方法是否修改未核查**，下述细节只归于预印本。

其数据为巴西、厄瓜多尔共 95 个 SIRGAS 站点的周 GNSS 三分量，使用 TimesNet、Kalman、Gaussian Process、Moving Average、LSTM 等方法进行异常分数对比；在 BRAZ/CHEC 两站优化超参数并比较跨站点应用。预印本 Data preprocessing 将记录断点标 1，其余标 0，并把缺口用 0 填充、缺口首点及缺口后首点加入异常标签。它采用二元异常标签，没有在该协议里将持续运动与质量形态作为可并存两轴标签。

它报告 Top 1% 与 Top S，后者 S 明确来自该站真值异常总数。**Top S 属于真值辅助的排序诊断**，不能视为未知异常数量条件下的部署结果。该文能作为实际 GNSS 数值/机器学习近邻，但不能把其标签、插补与 oracle 选择无说明地迁入本分支。

论文明确给出作者代码 [mauriciodev/deep_anomaly_gnss](https://github.com/mauriciodev/deep_anomaly_gnss)。本轮 README 抓取遇到 DNS 失败，`git ls-remote` 被现有本地代理端口连接失败阻断，未取得代码 SHA；因此这里核实的是“论文声明的代码 URL”，不是已完成源码复现或直接可执行的保证。

### 6.4 对当前研究表述的约束

现有证据已经覆盖“正常参考视觉分类”“视觉/数值互补”“GNSS 趋势、断点、异常值分别分析”和“真实多站点 GNSS 数值学习”四类近邻。当前可检验的问题是：在统一不看测试真值的协议下，**允许运动与质量形态并存的 GNSS 标签、原始位移活动边界约束、站点正常参考和单轮数值/视觉融合**是否改善错误类型与定位性能，并且能否跨站点尺度迁移；不是证明这些组合首次出现。

在本轮三项工作及先前已读取来源里，**尚未确认与上述完整标签和证据契约完全相同的方案**。这项状态应写为“有界检索下未确认”，不得改写成“首次”“研究空白已经证明”或“创新性成立”。未深入核查搜索结果中其他候选的全文，未检索全部索引、中文 GNSS/滑坡文献或该方向后续引用链；正式投稿前仍需扩大范围并核查正式版及可获取的源码。

## EXP03继续核查：阶跃定位与GNSS TimesNet作者代码

2026-10-12重新阅读Crocetti2021正文3.1–3.4、4.4（缓存全文有哈希）。
21日重叠三分量原坐标窗口、窗内多类断点时间与缺测窗剔除，与本项目
48项局部特征的日级多标签RF不同。其3.3单独区分类型正确但时间错误，
支持本轮精确日和±3日辅助分别报告。3.4清洗有经验5倍噪声尺度，4.4比较5–15mm
断点筛选；本项目不迁入这些经验/毫米阈值，也不冒称其完整源码复现。

此前6.3仓库访问失败的历史记录保留。此次通过HTTP实际取得
[deep_anomaly_gnss](https://github.com/mauriciodev/deep_anomaly_gnss)提交
`7bf4e9f31215fa93265d7b59e6fcd744924d163d`的文件树、TimesNet模型、
StationTrainer、批处理及CHEC参数，另保存读取时的README/提交API响应。
来源与内容哈希见证据清单。实际读到：

- StationTrainer.train对同站training_data执行fit及decision_function，MinMax缩放后
  采用默认99百分位截点；use_du=False时选择N/E两列。
- process_stations默认use_du=False；其TopK数量直接读取truth中1的数量，
  属于使用真值异常总量的排序诊断，不能作为未知事件数下的部署结果。
- model/timesnet.py使用DeepOD的BaseDeepAD及工具、PyTorch；作者实现添加训练损失的
  early stop和权重恢复。CHEC参数seq_len=10、e_layers=2、d_model=d_ff=128、
  dropout=.2、top_k=3；这不是本轮日NEU任务已经验证的设置。

这些事实来自当前作者提交，不断言与正式出版正文每项一致。本轮尚未执行TimesNet，
不填对比数值；若后续迁移，需要将同站无监督拟合、训练跨站监督、
日/周频率、是否使用U、缺测、百分位和oracle TopK分别声明。

## EXP04继续核查：形态分工与F1定义（2026-10-13）

本轮重新实读已取得并有哈希的三篇全文，未宣称完成新的全库检索。

- [VisualTimeAnomaly / Can Multimodal LLMs Perform Time Series Anomaly Detection?](https://arxiv.org/abs/2502.17812)
  的已缓存版本§2使用按类型注入的简单合成时序，报告precision、recall与F1；
  §6.2–6.3明确用传统方法处理point、用视觉模型处理range，并有checking修正。
  因此EXP04“尖峰数值、阶跃排名视觉、持续形变视觉边界”的工作是GNSS细化和实证，
  不能声称发明了按粒度分工或数值/视觉互补。作者完整代理代码未在已取得公开目录中确认。
- [VLM4TS](https://arxiv.org/abs/2506.06836)已缓存全文的附录
  “Data Preprocessing, Postprocessing and Evaluation”明示扫描
  alpha∈{0.10,0.01,0.001}并报告F1-max。其unweighted contextual定义按预测区间
  是否有任意交集计TP/FP、按真区间是否无交集计FN，不等同本项目一对一IoU匹配。
  不将原文F1-max直接作为本项目固定判决下的比较分数。
- [TAMA](https://arxiv.org/abs/2411.02465)已缓存v1 Table 1、Metrics段及Appendix A.2
  明示PA-F1，并承认PA可能高估性能。本轮保留无PA的日级、事件级及时间容差指标，
  允许论文选择效果较好且有领域意义的指标，但说明是在验证集选择，并给完整矩阵。

指标选择服务于“多形态识别”这一目标：逐类汇总F1再四类等权，避免长持续活动在
通用逐日micro-F1中压过稀疏质量异常。它是标准宏平均应用，不包装为新的评价指标。
±3日只评价点事件定位容差；不改标签、不吸附预测日期、不用于选择数值判决阈值。

## 参考文献

1. Xu, X., Wang, H., Liang, Y., Yu, P. S., Zhao, Y., & Shu, K. (2026). Can Multimodal LLMs Perform Time Series Anomaly Detection? *Proceedings of the ACM Web Conference 2026*. https://doi.org/10.1145/3774904.3792376
2. He, Z., Alnegheimish, S., & Reimherr, M. (2025). Harnessing Vision-Language Models for Time Series Anomaly Detection. *arXiv*. https://arxiv.org/abs/2506.06836
3. Crocetti, L., Schartner, M., & Soja, B. (2021). Discontinuity Detection in GNSS Station Coordinate Time Series Using Machine Learning. *Remote Sensing, 13*(19), 3906. https://doi.org/10.3390/rs13193906
4. Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing, 167*, 107299. https://doi.org/10.1016/j.sigpro.2019.107299
5. Zhuang, J., Yan, L., Zhang, Z., Wang, R., Zhang, J., & Gu, Y. (2024). See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers. *arXiv*, 2411.02465v1. https://arxiv.org/abs/2411.02465
6. Heflin, M., Donnellan, A., Parker, J., Lyzenga, G., Moore, A., Ludwig, L. G., Rundle, J., Wang, J., & Pierce, M. (2020). Automated Estimation and Tools to Extract Positions, Velocities, Breaks, and Seasonal Terms From Daily GNSS Measurements: Illuminating Nonlinear Salton Trough Deformation. *Earth and Space Science*. https://doi.org/10.1029/2019EA000644
7. Andrade, L. C. O. de, Paulo, M. C. M. de, & Marques, H. A. (2025). Anomaly detection in weekly SIRGAS GNSS time series: a TimesNet assessment in real data. *GPS Solutions*. https://doi.org/10.1007/s10291-025-01839-3 （本轮方法阅读来自其 2024 Research Square v1 预印本：https://doi.org/10.21203/rs.3.rs-4460921/v1。）
