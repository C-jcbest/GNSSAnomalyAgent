# 基于持续性配对视觉—数值学习的GNSS形变异常识别及多形态验证

> 中文完整研究初稿，2026-10-12。本稿以整体形变异常为主，弱运动与A/S/D为附加分析。结果来自已完成实验；新增多形态评价属于方法开发，保留测试尚未打开。旧v54稿保持原样。本稿尚未达到算法创新论文的投稿证据要求。

## Material Passport

- Origin Skill: academic-research-suite；用户授权的研究、生成数据、实验与写作。
- Version Label: deformation-manuscript-20261012-v1。
- Verification Status: DQ-VN完整模型/概率/评分重放通过；PP-VN及逐轴12选中模型、210,240个概率、72条逐轴证据及118个训练反事实重构通过，24输出组完整评分一致。
- Reference Status: 受控生成过程参考；已有Codex原图与数值复核为开发者辅助参考，没有独立领域专家或现场总活动金标准。

## 摘要

长期GNSS形变识别需要区分连续运动与仪器偏移、离群尖峰、活动后平台及观测缺口。本文研究持续性配对视觉—数值学习（PP-VN）：在受控训练侧将连续形变替换为同净位移的一次偏移，保留原噪声、缺测及已有观测干扰，利用冻结图文预训练模型的视觉分支和数值特征学习活动识别。配对干预在既有24条新受控记录上相对同增广提高活动F1约0.0225，但扩展研究需要检验更丰富的形态与干扰组合。为此新增96条730日记录，覆盖静稳、仅偏移、仅尖峰、单次形变、快速连续形变、再活动和两类混合干扰，按完整记录划分48训练、24验证和24保留评价。规则、PELT和XGBoost在相同训练／验证划分重新校准或训练，并开展去配对、打乱配对、去视觉、去数值及辅助质量监督消融。多形态验证中，PP-VN活动F1为0.7730，略高于同增广0.7713，但新增82个误报日、补检56日，事件F1从0.7368降至0.6667；同增广强XGBoost活动F1为0.7915。形变／质量双输出和逐轴持续性证据也未获得稳定整体收益。结果支持配对干预在特定受控分布中的价值，同时否定其已具备广泛形变识别优势的结论。本文公开完整负结果、缺测与评分契约、结果重放及推广限制，不将辅助阶段收益或Agent框架包装为异常识别创新。

**关键词：** GNSS；形变异常；观测干扰；持续性配对；视觉数值融合；事件识别。

## Abstract

GNSS deformation detection must distinguish continuous motion from instrument offsets, spikes, post-motion plateaus and missing observations. We investigate paired-persistence visual–numeric learning (PP-VN), which replaces controlled training deformation with an equal-displacement offset while preserving noise, missing dates and existing observational artifacts. A frozen pretrained vision branch and numerical features feed a small activity classifier. Following a narrower controlled confirmation, we generated 96 diverse 730-day records and assigned complete records to training, validation and reserved evaluation sets of 48, 24 and 24 records. Numerical rules, PELT and XGBoost were refitted or recalibrated on the same splits. On the exposed diverse validation set, PP-VN achieves daily F1 of 0.7730 versus 0.7713 for identical augmentation without the pair loss, but introduces 82 false-positive days while recovering 56 positive days. Event F1 decreases from 0.7368 to 0.6667, and same-augmentation XGBoost achieves daily F1 of 0.7915. Auxiliary deformation/quality supervision and componentwise persistence evidence also fail their predefined development criteria. The findings bound the usefulness of pairing rather than establish broad or field-level superiority. Reserved evaluation references remain unopened; developer-assisted observation review is distinguished from independent expert ground truth.

## 1 引言

长期坡体形变可以呈现持续、间歇和速度变化等形式。GNSS累计位移同时受到相关误差、仪器偏移与离群观测影响。将累计位移保持高位判为持续活动，会把已停止的运动或设备偏移后的平台误识别为形变；只使用局部导数，又可能在真实起变之前产生响应。因此，本文主任务是识别目标形变活动及其区间，首先评价Precision、Recall、F1、事件匹配和边界，随后才分析活动内部的加速、近匀速与减速。

视觉模型能够利用曲线形态，但观察范围、噪声缩放和目标日期的对应关系会影响其判定。已有视觉与语言时间序列方法已经采用多尺度观察、候选复核及视觉数值融合[1–3]。仅将这些步骤串成Agent，或增加提示和工具次数，不能构成本文的方法贡献。本研究选择一个可隔离的训练干预：保持首末位移、原观测误差与缺测基本相同，仅改变期间是否持续运动，检验模型能否学习额外的持续性差异。

本文报告三个可核查的研究工作：一是同净位移持续性配对方法及模态、配对关系消融；二是形变与质量干扰可以共存的多形态受控数据和统一强基线；三是对已有受控收益的推广检验。第三项得到负结果，故这些工作目前属于方法与适用边界研究，不能合并表述为已获得整体GNSS异常识别优势。

## 2 相关工作与技术差异

VLM4TS将视觉候选检测与长上下文VLM修订结合，并采用多尺度及正常参照[1]；Refining Time Series Anomaly Detectors using Large Language Models研究数值检测后的多模态误报复核[2]。这些研究说明组合识别具有可研究性，也意味着“先数值后视觉”已经有最近邻方法。本文不能将通用复核流程称为首创。

VETime研究视觉与时间轴对应及数值融合[3]。本文明确每个图像中心对应原日历中心，但没有实现VETime的完整网络或复现作者实验。CATCH研究选择性通道相关建模[4]，启发了本项目的逐轴证据预检；本次没有运行CATCH，不能把一个逐轴特征扩展称为CATCH复现。

PELT[5]和XGBoost[6]分别提供变化点与监督学习强基线。本文将它们的输出统一转换为活动及活动内阶段，不把变化点个数直接当形变事件检出。冻结MobileCLIP2视觉编码[7]是已有预训练能力，本文没有训练新的基础模型。CAPS已有配对、打乱对照与排序目标[8]，因此配对或softplus损失本身不新。可能的技术差异仅限于GNSS同误差、同净位移的持续运动／偏移训练干预及其实际识别收益，是否形成可发表的新贡献还需要更全面最近邻核查与未参与选择的数据验证。

## 3 任务与方法

### 3.1 形变与干扰的非互斥定义

输入为原日历上的三维位移 $x_t\in\mathbb R^3$ 与可用性 $o_t$。主标签 $a_t\in\{-1,0,1\}$ 表示未知、非目标形变及目标形变。仪器偏移转换或尖峰另记质量事件 $q_t$，形变与质量事件可以同时阳性。永久偏移后的平台不连续标为质量事件，也不自动标为形变。

$$
x_t=g_t+j_t+s_t+e_t,
\qquad
g_t=\sum_{u\le t}v_u d.
$$

$g_t$ 为连续形变，$j_t$ 为观测偏移，$s_t$ 为尖峰，$e_t$ 为测量误差，$d$ 为三维方向。本分解只在受控生成和训练干预中可用，检测器在评价时只能读取观测。缺测保持未知；确定阶段只能位于已确认活动内部：

$$
o_t=0\Rightarrow\widehat a_t=\widehat s_t=-1,
\qquad
\widehat s_t\in\{A,S,D\}\Rightarrow o_t=1,\ \widehat a_t=1.
$$

当前任务将设计的坡体形变作为阳性。其他场景若存在正常长期趋势，需要重新定义目标，不能将本文所有持续运动直接移作异常。瞬时真实位移与仪器偏移可能具有相同单站观测，本文不保证由NEU曲线恢复唯一成因。

### 3.2 原日期视觉与数值输入

原始局部图由61个三日槽构成，覆盖中心前后约90日。每槽至少两天完整三轴观测才画中位数；缺测断线并显示掩码。三轴分别居中，纵向尺度至少为各轴六倍噪声代理，避免任意放大噪声。图像中心标记保持原时间对应，不压缩日历。

$$
z_t^V=\frac{E_{\mathrm{frozen}}(I_t)}{\|E_{\mathrm{frozen}}(I_t)\|_2},
\qquad
p_t=\sigma\!\left(f_\theta([z_t^V,z_t^N])\right).
$$

冻结MobileCLIP2-S0/dfndr2b视觉分支产生512维编码。数值输入为已有18项原位移／诊断特征与18个不可用指示，归一化统计仅来自合法训练中心。活动头为16或32单元ReLU隐层。模型使用未来和过去语境，属于离线回溯识别，不能称实时预警。冻结图文模型的视觉分支也不等于新语言生成或Agent推理。

### 3.3 同净位移持续性配对

对训练中心 $c$，取端点 $u=c-91$、$v=c+91$。配对负观测移除训练形变分量，替换为一次偏移：

$$
\widetilde x_t=x_t-g_t+g_u+\mathbf1_{t\ge k}(g_v-g_u),
\qquad
k\in\{c-42,c,c+42\}.
$$

端点直接复制原观测，使两端数值精确一致。由于只移除 $g_t$，原测量误差、已有仪器偏移、尖峰及缺测均保留；负例可以存在多个设备干扰，不等于理想平直背景。中心原14日查询块需至少10日有观测，且所有确定参考均为活动，端点有观测并满足原输入支撑。28日网格下取最近的3日训练采样中心，所有合法配对保留，不以幅度或结果筛选。

新增数据中形成118对，来自25个训练parent，覆盖五种形变族。配对负中心不是118条新独立记录；同一parent及其后代始终属于训练组。只在训练侧使用已知形变分量，不读取验证潜在分量生成配对，更不接触保留评价过程。

$$
\mathcal L_{\mathrm{pair}}=
\frac1{|\mathcal P|}\sum_{(c^+,c^-)\in\mathcal P}
\log\!\left(1+\exp(1-z_{c^+}+z_{c^-})\right),
\qquad
\mathcal L=\mathcal L_{\mathrm{BCE}}+\mathcal L_{\mathrm{pair}}.
$$

原样本与负中心共享带正类权重的BCE，每个监督minibatch另计算全配对平均排序项。固定seed20261020、12epoch、Adam学习率0.001、weight decay0.0001、batch128、配对系数和margin均为1。不因本次验证结果改变损失、种子或画法。

### 3.4 两个被否定的最小机制

DQ-VN在相同视觉数值共享表示上增加独立质量查询输出，活动与质量非互斥。质量标签表示中央14日内是否存在已观测干扰；至少8日支持，否则未知。

$$
(p_t^{\mathrm{def}},p_t^{\mathrm{quality}})
=(\sigma(w_d^\top h_t),\sigma(w_q^\top h_t)),
\qquad
\mathcal L_{\mathrm{DQ}}=\mathcal L_{\mathrm{def}}^{\mathrm{balanced}}+
\mathcal L_{\mathrm{quality}}^{\mathrm{balanced}}.
$$

单输出消融关闭质量损失，网络、样本和初始化相同。逐轴证据预检则在21、42、84日尺度比较三个非重叠14日原位移块。对轴 $a$ 定义：

$$
D_{t,a}^{(h)}=
\operatorname{sign}(\Delta^-_{t,a}\Delta^+_{t,a})
\frac{\min(|\Delta^-_{t,a}|,|\Delta^+_{t,a}|)}{\widehat\sigma_a}.
$$

仅增加9个描述量，池化消融获得完全相同原块后丢弃轴身份。它不是活动标签、概率或显著性检验。这两项机制均未通过预设条件，作为研究负结果保留，没有扩大为视觉Agent或增加模块挽救。

## 4 数据与实验协议

### 4.1 多形态数据与参考

新增96条、每条730日，共70,080个原日历位置，分48训练、24验证、24保留评价。八族在每组分别为6、3、3条：静稳、仅永久偏移、仅尖峰、普通单次形变、快速连续形变、停滞后再活动、形变叠加偏移、形变叠加尖峰。普通活动速度形态可为恒速、上升、下降、圆滑峰或不规则变化，不强制A→S→D顺序。

复用已有stable模拟器的measurement_noise及原缺测，剔除其原运动与原干扰后，使用独立随机流注入本轮分量。普通活动35—259日、峰值0.035—0.8 mm/日，快速活动8—35日、峰值1—3.5 mm/日；参数用于覆盖设计，没有现场发生率或设备噪声标定。混合场景首个干扰位于活动内，恰逢缺测不重抽。训练／验证参考有29个已观测形变与质量同时阳性日。

受控参考定义设计正速度日为活动，缺测标签为−1。它能检验运动过程恢复，但微弱运动起止未必能从噪声观测唯一确定。已有Codex视觉及数值辅助标签只能作开发者观测敏感性，不是独立领域专家参考。当前保留24条参考内容没有用于拟合、选方法、调整阈值或评分；因为三个新版本均未通过开发门槛，没有打开它们救分。

### 4.2 公平强基线与消融

规则、PELT、原XGBoost、raw84和chart384在相同48／24上重新校准或训练。新学习臂均有2个宽度／深度与19个阈值，共38次机会，按日F1加事件F1选首个最大值。原规则／PELT沿用合理网格重新校准。神经模型训练中心每3日抽样；强树使用全部合法训练日期，明确保留其覆盖优势，不把它隐去。

PP-VN七臂分别是：主配对、同增广去配对、无增广普通头、打乱配对、去数值、纯数值配对和纯数值增广。主配对、去配对与打乱配对看到完全相同监督样本；raw84/chart384强树增加相同负中心。chart384还获得原图各列位置与可用性，避免把图提供的额外时间信息全部解释为视觉编码价值。

DQ五神经臂用于辅助质量监督和模态消融。逐轴小试有raw84、同块池化和保留轴身份三臂。所有新方法使用同一新训练数值阶段头，活动先确定再门控阶段。直接VLM与固定工作流本批尚未重新运行，不能将历史人口成绩填入本轮对比表；已有完整历史比较及费用仍保留在旧稿和原产物。

### 4.3 评分与继续条件

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},
\qquad F_1=\frac{2TP}{2TP+FP+FN}.
$$

对已知阳性，不可用预测仍计FN；参考未知不当作正常，也不从分母中删除已知阳性的弃判。事件按原评分器在可评价日历上进行一对一IoU匹配，报告0.3和0.5阈值，不使用point adjustment。起止MAE仅在匹配事件上计算，匹配集合变化时不能当同事件配对改善。所有短事件保留；原14日最短段后处理可能漏掉8—13日快速形变。

PP-VN扩展在拟合前写下八项继续条件：对同增广日F1严格增益、不低于打乱配对、对普通和同增广均净纠错为正、事件F1不低于同增广、纯干扰FP不高于普通头、混合召回不低于同增广、对纯数值配对有日F1增益或事件增益且日F1损失不超过0.01。全部通过才进入保留确认。DQ与逐轴小试另有原冻结六条件；失败结果完整保留。

24条验证记录已用于机制反馈，因此即使进行整记录配对bootstrap，也只能描述本组波动。2000次重采样不消除调参、方法选择及多重比较偏差，不能以大量逐日位置冒充独立样本量。

## 5 实验结果

### 5.1 多形态PP-VN及主要强对照

| 方法 | Precision | Recall | 活动F1 | 事件F1(.3) | 事件F1(.5) | FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PP-VN主配对 | 0.854409 | 0.705816 | 0.773036 | 0.666667 | 0.512821 | 213/521 |
| 同增广去配对 | 0.901132 | 0.674195 | 0.771318 | 0.736842 | 0.526316 | 131/577 |
| 无增广普通视觉数值头 | 0.916468 | 0.650480 | 0.760898 | 0.756757 | 0.594595 | 105/619 |
| 打乱配对 | 0.905932 | 0.603614 | 0.724500 | 0.628571 | 0.514286 | 111/702 |
| 配对去数值 | 0.687452 | 0.504235 | 0.581759 | 0.580645 | 0.516129 | 406/878 |
| 纯数值配对 | 0.878106 | 0.638622 | 0.739457 | 0.536585 | 0.439024 | 157/640 |
| 纯数值同增广 | 0.887925 | 0.693394 | 0.778694 | 0.682927 | 0.536585 | 155/543 |
| raw84 XGBoost同增广 | 0.918033 | 0.695652 | **0.791519** | 0.731707 | 0.536585 | 110/539 |
| chart384 XGBoost同增广 | 0.866307 | 0.724449 | 0.789053 | **0.789474** | 0.526316 | 198/488 |
| raw84 XGBoost原数据 | 0.935561 | 0.664032 | 0.776750 | **0.789474** | **0.631579** | 81/595 |
| 原XGBoost重新训练 | 0.889555 | 0.586674 | 0.707043 | 0.742857 | 0.514286 | 129/732 |
| 鲁棒规则重新校准 | 0.946210 | 0.437041 | 0.597914 | 0.620690 | 0.551724 | 44/997 |
| PELT重新校准 | 0.973109 | 0.326934 | 0.489434 | 0.615385 | 0.461538 | 16/1192 |

完整21项来源区分对照及八族分表另存[完整报告](../artifacts/gnss-deformation-diversity-2026-10-12/diverse-paired-report-v1/report.md)。主方法相对同增广提高F1仅0.001719，补检56日却新增82FP，净纠错−26；相对普通头净−10。混合召回从0.642667提高到0.657333，但纯干扰FP49→95。八项条件通过四项，决定STOP_DIVERSE_PPVN_CALIBRATION。

![多形态验证集活动与事件F1对比](D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/gnss-deformation-diversity-2026-10-12/comparison-figure-v1/daily-event-comparison.png)

图1：同一24条验证记录的日活动F1与事件F1。红色为PP-VN；所有工作点均按原冻结验证协议选择。该图属于方法开发结果，没有保留评价或独立确认含义；PDF同名版本另存，可用于排版。

主方法匹配起止MAE为11.462／23.615日，同增广为9.714／16.071日。匹配事件集合不同，只能作为定位代价描述。已实际查看case_0385原三轴及各预测：观测包含两个明显永久转换，转换两侧平台没有相称的持续位移积累；主配对仍将转换附近及后平台部分判为活动。此图是评分后误报分析，不是新的独立标签，未追改过程参考。

### 5.2 辅助质量监督与逐轴证据

| 机制／对照 | 活动F1 | 事件F1(.3) | 纯干扰FP | 混合召回 |
| --- | ---: | ---: | ---: | ---: |
| DQ视觉数值双输出 | 0.734428 | 0.764706 | 35 | 0.600000 |
| 同输入单输出 | 0.765252 | 0.777778 | 35 | 0.605333 |
| raw84逐轴小试对照 | 0.776750 | 0.789474 | 15 | 0.630667 |
| 同新原块池化证据 | 0.786714 | 0.756757 | 22 | 0.622667 |
| 保留逐轴证据 | 0.736512 | 0.764706 | 0 | 0.538667 |

DQ净纠错−68，六条件通过两项。固定0.5概率下，视觉数值质量查询Precision为0.087003、Recall为0.466411、F1为0.146651，说明辅助任务未学好。查询阳性含义为中央14日干扰存在，不能把它当精确故障定位指标。逐轴预检虽然消除本组纯干扰FP，却损失混合召回、净纠错−81，六条件仅通过一项。两者均停止，未增加昂贵视觉复核或事后选用更好的消融替代主方法。

### 5.3 历史受控收益及推广边界

此前冻结PP-VN在另一新24条较窄受控记录上得到活动F1 0.872191、同增广0.849705，原预设10条件全部通过；原XGBoost84为0.874752。随后120条完整受控比较中，PP-VN活动F1 0.908419，同增广0.892971，强XGBoost84为0.909885；PP-VN事件F1 0.657061低于同增广0.674847与强XGBoost84 0.694611。观测参考上的配对收益也没有同步成立。

这些结果来自不同冻结人口，不能同新多形态验证合并计算总体F1，或将跨人口差异归为单一机制效应。它们共同说明：配对排序能够改变召回与持续性区分，但收益受形态、干扰构成、参考和事件终点限制，不能仅依据弱召回提升证明整体异常识别价值。

### 5.4 阶段与计算代价

主PP-VN活动内阶段macro-F1为0.510970，端到端为0.475225；同增广分别0.470268与0.449906。所有方法使用共同数值阶段头，差异来自活动检出及评价支持，不能当新增视觉A/S/D推理收益。弱日召回0.451852高于同增广0.339506，但这是附加结果，不覆盖主事件和误报条件失败。

训练／验证原视觉编码25,586中心，CPU实测1375.669秒；118配对准备233.590秒，其中新增负图编码7.365秒。缓存消除了重复编码，但原视觉成本仍需报告。新头与树的训练和验证循环另存training-costs.csv，当前没有完整部署端到端延迟或能耗比较。此次没有新增远端VLM API调用。

## 6 讨论

### 6.1 配对改善为什么不足以构成整体收益

同净位移配对迫使模型区分持续运动与偏移，在旧受控分布及本组纯数值配对对照中体现了部分增量。但主方法提高召回的同时增加误报，且事件分数下降。永久偏移落入观察窗口的不同位置、真实形变与干扰同时存在、短事件及活动尾段，都可能使原配对负例不能充分约束目标日期的活动归属。当前结果只支持这些作为待验证解释，未单独隔离各原因。

从该结果出发，下一项创新需要作用于事件完整性、平台误报或目标日期证据归属，并与同信息强数值对照分离。不应再次只扩大弱召回，也不应增加更多输出头、复杂多智能体或提示搜索来掩盖失败。局部视觉复核只有在真实错误样例上展示新增纠错价值，才值得纳入完整方法。

### 6.2 可信参考与独立性

受控过程真值提供准确的注入支持，却不意味着每个起止日都可从噪声曲线确定。独立参考建设应区分可确认内部、起止待定和不可识别范围；不能因模型漏掉微弱尾段就将其重标为静稳，也不能因数值导数提前响应就确认活动。Codex实际视觉能力与辅助数值可以形成可追溯开发参考，但同一开发者知道方法、生成假设与历史结果，不能称独立专家。

保留评价能减少方法选择泄漏，但与训练共享模拟器，仍不能证明现场泛化。现场验证还需跨站点、跨观测背景划分与可信领域参考。当前没有外部复核者，本文如实保留该缺口，同时继续自主可完成的方法与仿真工作。

### 6.3 发表条件

目前能够形成完整研究稿及可复现实验，但尚不能据新增数据宣称一个已验证的整体Agent/VLM创新。算法创新投稿至少还需要明确机制相对固定观察、去数值反馈及强数值的稳定增益，未参与方法选择的新记录确认，以及更充分的最近邻方法比较。旧PP-VN局部分布收益与新的推广失败可以支撑适用边界讨论，不能替代这些证据。是否适合某期刊还取决于其定位、真实数据要求和研究问题，目前未选定投稿目标。

## 7 结论

本文以整体GNSS形变异常识别为目标，完成持续性配对视觉数值方法、多形态数据扩充、强基线重训及关键消融。配对方法在已有窄受控分布上具有机制收益，但在新增多形态开发数据上，日F1的小幅增益伴随更高误报和更低事件F1，未通过预设推广条件。辅助质量监督和逐轴证据也没有解决这一问题。当前证据支持保留配对方法的受控价值和负结果，继续围绕事件与平台区分开发一个更有效的小机制，而不宣称已获得广泛视觉或Agent优势。

## 数据、代码与AI使用说明

新数据协议、train／validation参考、输入、生成分量与保留参考封存位于`artifacts/gnss-deformation-diversity-2026-10-12/data-v1`。DQ、逐轴和配对目录分别保存全部网格、选中模型、概率、原日历活动／阶段结果、分组与费用。复现入口为`prepare_landslide_deformation_diversity.py`、`encode_landslide_deformation_quality.py`、`train_landslide_deformation_quality.py`、`probe_landslide_axis_evidence.py`、`prepare_landslide_diverse_pairs.py`、`train_landslide_diverse_pairs.py`及对应verify/report脚本。保留评价文件只核验哈希，不在本稿工作中加载内容。

Codex参与文献定位、代码、受控数据生成、结果分析、原图检查和稿件写作；生成过程参考不等于AI看图标注，实际看图记录与历史AI开发参考另存。文献支持范围及源码读取记录见[来源笔记](../artifacts/gnss-deformation-anomaly-research-2026-10-12/reading-notes.md)。没有宣称由独立专家标注、复现所有作者模型或完成现场验证。

## 参考文献

下列条目使用本轮已核查论文标识与链接，未补造作者、卷页或未核实会务信息。正式投稿前按目标格式从出版者导出完整书目。

[1] Harnessing Vision-Language Models for Time Series Anomaly Detection（VLM4TS）. [arXiv:2506.06836](https://arxiv.org/abs/2506.06836)；AAAI登记[DOI:10.1609/aaai.v40i26.39319](https://doi.org/10.1609/aaai.v40i26.39319). [公开代码](https://github.com/ZLHe0/VLM4TS).

[2] Refining Time Series Anomaly Detectors using Large Language Models. [arXiv:2503.21833](https://arxiv.org/abs/2503.21833), 2025.

[3] VETime. [arXiv:2602.16681](https://arxiv.org/abs/2602.16681), 2026. [公开代码](https://github.com/yyyangcoder/VETime). 此处按已核查预印本引用。

[4] CATCH. [arXiv:2410.12261](https://arxiv.org/abs/2410.12261). [公开代码](https://github.com/decisionintelligence/CATCH).

[5] Killick R, Fearnhead P, Eckley I A. Optimal Detection of Changepoints With a Linear Computational Cost. Journal of the American Statistical Association, 2012. [DOI:10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745).

[6] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD, 2016. [DOI:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).

[7] Faghri F, Vasu P K A, Koc C, et al. MobileCLIP2: Improving Multi-Modal Reinforced Training. [arXiv:2508.20691](https://arxiv.org/abs/2508.20691), 2025. [官方源码](https://github.com/apple/ml-mobileclip). 核查范围为README、实际配置和载入视觉权重；未读论文全文或重训。本文使用S0/dfndr2b，权重SHA256、实际transform和环境版本保存在运行时及编码目录。

[8] Gao Y, Lan T, Lu Y, et al. Supervision Recovery for Time Series Anomaly Detection via Context-Anchored Pairing. [arXiv:2609.33610v1](https://arxiv.org/abs/2609.33610v1), 2026. [作者源码](https://github.com/gyf-0623/caps). 沿用旧v54实际核查的PDF方法段落、README及固定commit源码范围，未运行作者模型。
