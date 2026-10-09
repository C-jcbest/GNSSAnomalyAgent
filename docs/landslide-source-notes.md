# 长期滑坡位移仿真：来源核查笔记

核查日期：2026-10-07。用途：为本项目新一批长期 GNSS 位移**合成实验数据**提供现象依据，不将合成曲线称为现场实测，也不将设计参数称为现场标定值。本次为任务导向的定向检索，不是系统综述。

## 已核对的来源与支持范围

### 1. 库岸滑坡的阶梯位移

Li, Y., Huang, J., Jiang, S.-H., Huang, F., & Chang, Z. (2017). *A web-based GPS system for displacement monitoring and failure mechanism analysis of reservoir landslide*. Scientific Reports, 7, 17171. [DOI: 10.1038/s41598-017-17507-7](https://doi.org/10.1038/s41598-017-17507-7)。

- 实际访问并阅读：[出版社全文](https://www.nature.com/articles/s41598-017-17507-7)，HTTP 200；重点为摘要及 Fig. 9 附近的监测位移讨论。
- 核对内容：王庙滑坡前部 GPS 1 和中部 GPS 2 的位移存在阶梯特征，变化受季节性及周期性触发因素影响；2015—2017 年的两个变形周期结合库水位与降雨讨论。摘要说明快速降低的库水位可触发再活动；强降雨重要，但不是该案例的决定性因素。
- 简短依据摘录：“there are step-like characteristics in the landslide displacements on GPS 1 ... and GPS 2”。
- 可支持的设计：长期低速背景与若干有限持续时间的加速段叠加；段间重新变缓。不同位置、不同年份的幅度与响应应允许差异。
- 不能声称：所有阶梯位移均由降雨单独引起；每年在相同日期出现相同幅度；文中位移 RMSE < 5 mm 是所有 GNSS 设备的通用误差。
- 证据类型：VI，单滑坡监测案例；仅作现象依据。

### 2. 长期缓慢滑动可以季节性加速后减速

Finnegan, N. J., & Saffer, D. M. (2024). *Seasonal slow slip in landslides as a window into the frictional rheology of creeping shear zones*. Science Advances, 10, eadq9399. [DOI: 10.1126/sciadv.adq9399](https://doi.org/10.1126/sciadv.adq9399)。

- 实际访问并阅读：[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11482306/) 对应的 [Europe PMC 全文 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11482306/fullTextXML)，HTTP 200；出版社页面 HTTP 403。重点为 Introduction、Results、Materials and Methods 及 Fig. 2–3。
- 核对内容：Oak Ridge 与 Minor Creek 两处缓慢滑坡的速度和孔隙水压力观测支持速度强化摩擦解释。文中明确讨论季节性加速和减速，冬季速度与孔压呈非线性变化；Oak Ridge 的 GPS 分析覆盖 2016—2024 年。
- 简短依据摘录：“many slow landslides undergo seasonal cycles of acceleration and deceleration”。
- 对图表的直接启发：论文由日位移用 11 天窗口估算速度，理由是较短窗口放大 GPS 速度噪声，较长窗口会平滑掉真实变化。项目可展示原始观测以及清楚标注窗口的平滑趋势；窗口长度仍是本项目的显示选择。
- 不能声称：每次加速均意味着临滑；全部滑坡服从同一个速度强化模型。文中的快速降雨响应（<1 天）受特定水文条件约束，不能作为全体样本固定时滞。
- 证据类型：VI，两处滑坡的长期观测和物理解释；对所测地点有直接证据，对其他地点需检验。

### 3. 持续加速与失稳预测存在适用条件

Intrieri, E., & Gigli, G. (2016). *Landslide forecasting and factors influencing predictability*. Natural Hazards and Earth System Sciences, 16, 2501–2510. [DOI: 10.5194/nhess-16-2501-2016](https://doi.org/10.5194/nhess-16-2501-2016)。

- 实际访问并阅读：[出版社文章页](https://nhess.copernicus.org/articles/16/2501/2016/) 与 [全文 XML](https://nhess.copernicus.org/articles/16/2501/2016/nhess-16-2501-2016.xml)，均 HTTP 200；重点为 Introduction 和 Discussion 的局限。
- 核对内容：对 15 次滑坡崩塌进行回溯分析。文中讨论依赖蠕变假设的加速及逆速度方法，并明确承认这些方法并非总能产生良好结果；自然或仪器噪声可阻碍预测，三级蠕变可能快到没有足够处置时间。
- 简短依据摘录：“such methods do not always produce good results”。
- 可支持的设计：纳入先缓慢累积、后持续加速的曲线；增加加速过程中短时减速和不规则波动，作为加速识别实验。另设加速后回稳样本，检验是否把暂时加速误当成持续恶化。
- 不能声称：某条合成曲线的速度或加速度阈值可直接作为真实预警标准；存在加速即可确定失稳时刻；模拟必须精确遵循反双曲函数。
- 证据类型：IV，历史事件回溯系列；不能替代前瞻性的预警性能验证。

### 4. GNSS 误差不能只用独立白噪声表示

Williams, S. D. P., Bock, Y., Fang, P., Jamason, P., Nikolaidis, R. M., Prawirodirdjo, L., Miller, M., & Johnson, D. J. (2004). *Error analysis of continuous GPS position time series*. Journal of Geophysical Research: Solid Earth, 109. [DOI: 10.1029/2003JB002741](https://doi.org/10.1029/2003JB002741)。

- 实际访问并核对：[Crossref 的出版者登记元数据与完整摘要](https://api.crossref.org/works/10.1029/2003JB002741)；出版社/DOI 跳转页 HTTP 403，**未核对全文**。
- 摘要支持：分析 414 个站的 954 条连续 GPS 序列；全球站网解的噪声可由白噪声与闪烁噪声组合较好描述；去除空间相关的共模信号后，区域解的噪声显著降低。
- 简短依据摘录：“a combination of white noise plus flicker noise”。
- 可支持的设计：独立测量扰动之外保留时间相关误差及较慢起伏，站间共享一部分误差也具有研究动机。
- 不能声称：简单 AR(1) 就等于闪烁噪声；本文给出了本项目接收机、采样率、环境的误差标定；随意设定的随机游走幅度符合该论文全部结果。
- 证据类型：IV，多站长期序列分析；当前核查覆盖限于摘要。

### 5. 年际强迫不同会导致再激活与速度不同

Handwerger, A. L., Fielding, E. J., Huang, M.-H., Bennett, G. L., Liang, C., & Schulz, W. H. (2019). *Widespread Initiation, Reactivation, and Acceleration of Landslides in the Northern California Coast Ranges due to Extreme Rainfall*. Journal of Geophysical Research: Earth Surface, 124. [DOI: 10.1029/2019JF005035](https://doi.org/10.1029/2019JF005035)。

- 实际访问并核对：[Crossref 的出版者登记元数据与完整摘要](https://api.crossref.org/works/10.1029/2019JF005035)；出版社/DOI 跳转页 HTTP 403，**未核对全文**。
- 摘要支持：2016—2018 年通过航空雷达干涉及像素跟踪观测加州北部大量慢滑坡，2017 年极端降雨对应更多活动滑坡；所比较的 51 个滑坡中有 49 个在 2017 年比 2016 年速度更快。
- 可支持的设计：不能让年周期强度和活跃月份完全重复；同一背景下不同滑坡也应有不同敏感性、再活动与持续活动行为。
- 不能声称：这些数据是 GNSS；文中的活动数量或比例应被当成仿真类别比例；将加州年际降雨结论直接迁移成三峡水库调度机制。
- 证据类型：IV，多地点跨年观测；当前核查覆盖限于摘要。

## 核查记录与限制

全部五项 DOI 的题名、作者与年份均用 Crossref 核对，并实际调用 Semantic Scholar DOI 接口匹配题名与年份。对应 paperId：

| 来源 | Semantic Scholar paperId | 内容核查范围 |
| --- | --- | --- |
| Li et al., 2017 | `7cf070131454de23582503ee2d71c8a95235a75e` | 出版社全文相关段落 |
| Finnegan & Saffer, 2024 | `ef332476d9fdf7f43e19f37219945a7a18164969` | 开放仓库全文相关段落 |
| Intrieri & Gigli, 2016 | `4bef1ce34db9a0358e72817f56bd265760807052` | 出版社全文相关段落 |
| Williams et al., 2004 | `661c6d34ffefff3d2a02828f7bcb84d76c45d1fd` | 登记摘要 |
| Handwerger et al., 2019 | `0e4594d465e56f87a552a39bbcb1b4d2c1b85278` | 登记摘要 |

以上核查证明书目存在与具体内容支持范围，不是撤稿、利益冲突或出版质量的全面认证。没有检索 Cabell、Scopus 或 WoS，也未完成逐项 COI 审核；不报告这些未做检查为通过。证据等级仅粗略标示研究设计，临床分级体系不能替代地学案例的场地适用性判断。

## 给实现者的设计建议（建议本身不是文献测定值）

1. 将潜在形变与测量过程分开：先生成连续的速度变化并积分得到形变，再叠加独立误差、相关误差、少量异常观测与缺测。保存潜在真值用于实验评估，但展示时必须标成“仿真真值”，不能当观测平滑线。
2. 覆盖低速蠕变、季节阶梯、长期加速、暂时加速后回稳、间歇再激活及稳定对照。一个季节内可有多个不等幅脉冲；脉冲前后缓变，不要求整点突然换挡。
3. “不机械”应落实在过程上：对激活时间、增长速度、持续长度、衰减速度和年际强度分别随机化，并使用时间相关的扰动。不要仅对完全相同的线性/正弦模板添加很大的白噪声。
4. 时间长度、采样间隔、噪声幅度、事件持续时间、总位移范围、缺测率等均为工程设计假设。可先用 1–3 年日尺度或更高频采样、毫米至数百毫米总位移的候选范围做可视化与算法测试；这些范围不构成真实滑坡总体分布，也不是临滑阈值。
5. 缺测和仪器跳变可以作为观测质量压力测试，必须与形变异常标签区分。本次所核查的五个来源没有给出可直接用于本项目的缺测率或跳变率，不能据此假装现场标定。
6. 图表保留较淡的原始观测，叠加明确标识的趋势线；趋势轴以全时段幅度、噪声尺度及明确的最小显示跨度留白。允许用户选择统一量程或单条适配量程，避免逐图自动缩放把稳定噪声放成显著形变；若采用截断或分位数轴域，必须让被裁切点可见并可恢复完整量程。
7. 批次验收关注类别是否完整、趋势是否可辨、跨样本事件日期/幅度/长度是否多样、观测误差是否保留相关性、缺测是否仍表示缺测，以及标签是否泄露给待测模型。不能仅凭图像“看起来真实”宣称已验证现实性。
