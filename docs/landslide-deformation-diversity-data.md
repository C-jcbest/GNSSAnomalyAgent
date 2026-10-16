# 整体形变与质量干扰：新增多形态数据及最小双输出实验

> **执行更新**：96条已生成，DQ-VN完整校准、逐轴证据预检和PP-VN多形态扩展均已完成，并因预设条件未通过而停止本版。DQ主F1 .734428低于单头.765252；逐轴.736512低于raw84 .776750；PP-VN .773036略高于同增广.771318，但事件F1下降、净纠错−26、纯干扰FP49→95。同增广raw84 XGBoost为.791519。完整结果与复现见[报告](../artifacts/gnss-deformation-diversity-2026-10-12/diverse-paired-report-v1/report.md)和[新稿](gnss-deformation-manuscript-draft-2026-10-12.md)。24保留参考内容仍未打开。下文候选设计描述保留提出时语境，不能覆盖实际停止决定。

## Material Passport

- Origin Skill: academic-research-suite；用户授权的方法开发、生成与实验
- Origin Date: 2026-10-12
- Verification Status: 96条输入/诊断及72条训练验证参考已核查，4条生成逐值重放；保留评价参考仅核验文件哈希，尚未评分
- Version Label: deformation-diversity-v1 / DQ-VN候选；无新增付费推理

## 新增规模与任务

新增 **96条 × 730天 = 70,080个原日历位置**，case_0337—case_0432。48训练、24验证、24保留评价；8族在各组分别6、3、3条。所有随机流和分配在生成前冻结，所有记录保留，不挑曲线、补抽样或替换难例。此次是新的受控分布，不是实测扩展。

| 场景 | 形变标签 | 单独的质量事件标签 | 覆盖意义 |
| --- | --- | --- | --- |
| 静稳 | 阴性 | 无显式注入事件 | 相关噪声与缺测误报 |
| 仅永久偏移 | 阴性 | 转换原日 | 防止偏移及后平台被当作形变 |
| 仅尖峰 | 阴性 | 尖峰原日 | 防止离群点被当作运动 |
| 普通单次形变 | 活动原区间 | 无显式注入事件 | 多幅度、时长、内部形态 |
| 快速连续形变 | 8—35日活动 | 无显式注入事件 | 连续快变与瞬时偏移区分 |
| 停滞后再活动 | 两个分开的活动区间 | 无显式注入事件 | 不跨平台合并事件 |
| 形变＋偏移 | 活动原区间 | 转换原日，可同时阳性 | 避免检测到偏移便否决真实运动 |
| 形变＋尖峰 | 活动原区间 | 尖峰原日，可同时阳性 | 混合观测识别 |

弱运动是幅度子组，A/S/D是活动内部过程附加标签，均不决定主数据构成或主终点。活动可以呈恒速、仅增速、仅减速、圆滑峰或不规则重复变化，不强制A→S→D。旧`seasonal_steps`仍代表目标季节滑动，本轮没有把旧阳性改为正常。

## 可追溯生成与参考

分别保存连续位移、观测误差、永久偏移/尖峰及原缺测日历：

$$
x_t=g_t+j_t+s_t+e_t,
\qquad
g_t=\sum_{u\le t}v_u d.
$$

复用已有`stable`模拟器的测量误差与缺测，排除其原运动/原质量干扰，再通过独立随机流注入本轮明确事件。误差包含原有白噪声、有色误差及短相关分量，参数没有现场标定。原位移不回零、不重拉幅度、不插值补缺测。

普通活动时长35—259日、峰值按0.035—0.8 mm/日的对数均匀设计范围抽取；快速活动8—35日、1—3.5 mm/日；再活动之间至少50日停滞。每例可有1—2次8—40 mm三维偏移，或2—6个按白噪声尺度6—20倍注入的尖峰。混合场景首个干扰位于活动内部；若该日恰好缺测，保留缺测，不重新抽样。参数是覆盖假设，不是文献估计的发生率或预警阈值。

主参考为已知合成过程：正设计速度定义活动，阶段仅在活动内部；质量事件的正式位置是偏移发生日或尖峰日，永久偏移后的平台不重复标为质量事件。缺测两类标签均为−1。质量分量和运动标签可以同时阳性。

这个过程参考能验证机制恢复，但不是独立领域专家对可观察运动的确认。后续Codex原位移＋辅助数值复核另存，不能冒充外部专家参考，也不能依据预测追改主标签。

## 候选方法及公平对照

候选暂名 **DQ-VN（Deformation-Quality Separated Visual-Numeric Learning，形变／质量分离视觉数值学习）**。最小模型、完整编码和校准已经完成；前8条训练记录的1690个中心实际训练冒烟完成12epoch，没有保存或选择该临时模型。完整结果以顶部更新和实际报告为准；此前“编码进行中、尚无效果结果”已失效。

复用冻结MobileCLIP2-S0图文预训练模型的视觉分支及原位移/诊断数值特征，以同一原时间坐标的局部图和数值输入共享一个小表示层，使用两个独立输出：

$$
h_t=f_\theta([z_t^{V},z_t^{N}]),\qquad
(p_t^{\mathrm{def}},p_t^{\mathrm{quality}})
=(\sigma(w_d^\top h_t),\sigma(w_q^\top h_t)).
$$

一个原日中心的辅助质量监督表示其中央14日查询块内是否含有**已观测**的质量事件；支持不足保留未知。正式质量事件日仍原样保存，不把辅助窗口范围当精确故障定位。两个输出非互斥，不用质量阳性硬否决活动。

$$
\mathcal L=\mathcal L_{\mathrm{def}}^{\mathrm{balanced}}
+\mathcal L_{\mathrm{quality}}^{\mathrm{balanced}},
$$

单输出消融仅关闭第二项损失；输入、抽样、主输出初始化、头宽及训练机会相同。该辅助监督是待验证干预，多任务本身已有研究，不能仅凭增加输出头称新贡献。时间对齐借鉴VETime的思路，此处没有实现作者完整模型或声称其复现。冻结视觉分支没有新的语言生成，也不称完整VLM推理Agent。

预定比较5个神经臂：视觉＋数值双/单输出、纯数值双/单输出、纯视觉单输出；同组重训规则、PELT、原XGBoost、raw84及完整图形信息XGBoost。新人口使用相同48/24划分重训或校准，旧模型仅作历史参照。神经模型训练中心间隔3日，强树模型使用完整支持训练日期，明确报告这项强数值额外覆盖优势。

每个新学习臂有38项相同宽度/深度及阈值机会，验证选择用活动日F1＋事件F1，不用阶段分数。保留现有14日最短段后处理，因此8—13日快速事件可能漏检；不删除这些参考或只报告较长事件，实际影响由分组结果说明。数值阶段头也在新训练组重训，活动先确定后门控阶段。

视觉双输出只有在验证集相对同数据单输出提高日F1、事件不退步、净纠错为正、纯干扰误报不增加、混合场景召回不下降，并具有相对纯数值的日或事件增益时，才进入保留评价。此门槛在拟合前写入编码协议；失败则停止该版本，不改种子/损失/阈值挽救。新主候选不用于复活已停止的NCDA单特征版。

## 当前实证与验证限度

此前[NCDA机制实验](../artifacts/gnss-ncda-mechanism-2026-10-12/report-v2/report.md)在24条暴露开发记录中日F1 .884790→.903065、事件F1 .636364→.707692，却增加静稳/偏移误报41→95，4/5继续条件，停止推广。其负结果、过程原图、消融及旧v54论文保持原样。

新增数据已经核查96条输入及288份诊断绑定；72条训练/验证的52,560日参考与分量逐值一致，有29个已观测的形变＋质量同时阳性日；4条生成逐值重放一致。保留24条标签内容尚未打开，文件哈希保留。12项新增针对性业务测试与Ruff通过。数据生成约119.17秒。

保留24条应在通过机制门槛、模型/阈值冻结后，先做预指定8条原观测Codex复核，再推理并统一评价全部24条；不把先看标签的生成过程当作独立专家盲评。当前已完成三轮实际机制对比，但尚无通过推广门槛的广分布候选，故继续封存，不提前称新方法有效或可发表。

## 产物与复现入口

- [生成协议](../artifacts/gnss-deformation-diversity-2026-10-12/data-v1/protocol.json)
- [数据验收记录](../artifacts/gnss-deformation-diversity-2026-10-12/data-verification-v1.json)
- [双输出实验冻结协议](../artifacts/gnss-deformation-diversity-2026-10-12/quality-encoding-v1/protocol.json)
- [来源及近邻依据](gnss-deformation-anomaly-method-directions-2026-10-12.md)

```powershell
$env:PYTHONPATH='artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-runtime-v1/site-packages;src;scripts'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/prepare_landslide_deformation_diversity.py --out artifacts/gnss-deformation-diversity-2026-10-12/user-data
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/encode_landslide_deformation_quality.py --data artifacts/gnss-deformation-diversity-2026-10-12/data-v1 --out artifacts/gnss-deformation-diversity-2026-10-12/user-encoding
# 原编码完成、encoding-freeze.json存在且核验后才执行：
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/train_landslide_deformation_quality.py --data artifacts/gnss-deformation-diversity-2026-10-12/data-v1 --encoding artifacts/gnss-deformation-diversity-2026-10-12/quality-encoding-v1 --out artifacts/gnss-deformation-diversity-2026-10-12/quality-calibration-v1
```

当前原编码进程已经启动，继续时先检查其实际进程句柄及输出，不能因暂时没有新日志而重启或重复编码。训练尚未启动；不能把模型定义、绿色测试或编码协议当作方法收益。
