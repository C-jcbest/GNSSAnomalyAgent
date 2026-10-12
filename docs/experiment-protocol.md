# GNSS 视觉数值异常识别实验协议 v4

实验编号：**GNSS-VLM-EXP01**。实验名称：**基于站点正常背景的GNSS视觉数值融合异常识别仿真对照与消融实验**。

此名称于2026-10-12完成实验后补充，作为管理标识；已保存运行中的冻结协议快照、登记哈希和实验条件保持原样。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + 用户授权实现与执行
- Origin Date: 2026-10-12
- Verification Status: 协议与实现；真实运行结果另记，仿真不保证现场有效性。
- Version Label: gnss-vlm-research-v4

## 问题与假设

在不同站点背景尺度、正常趋势和季节项存在时，三分量原始图与站点相对数值证据的单轮融合，能否比纯视觉及强数值方法更可靠地区分持续形变与质量形态异常，尤其是二者重叠时？视觉/数值融合与代理已有近邻，组合本身不是创新。领域任务约束及其效果仍待证伪，依据见[literature.md](literature.md)。

这是离线观测形态识别，不研究失稳时间、在线预警、周跳、欺骗或仪器故障归因。阶跃可能有不同物理成因。

## 数据与隔离

默认每站730日，前365日是正常历史背景；较短记录使用一半记录，但主实验不使用不足一年背景。小批训练/验证/测试各6站；主批各24站，使用另一种子。独立随机流，无共享站点、派生背景或重叠窗口跨集。固定种子是复现选择，仿真参数范围不是现场阈值或总体分布。

开发v1的24次请求有字段/阶段重叠契约失败，半年背景拟合年周期也有正常外推错误。v2明确必需字段、互斥阶段和JSON响应格式，以完整年背景估计年周期；旧请求与失败全部保留。此为正式测试前开发修正，不是测试调参。

v2仍有6/24响应漏字段或额外阶段字段。一次单独能力探测确认当前服务支持严格JSON Schema（HTTP200且契约通过），v3启用完整结构化输出；不削除字段或事后修补预测。增加局部偏离、绝对增量/斜率作为方向无关的共享数值证据，并单列rf_balanced应对稀少点标签；默认RF保留。这些为开发阶段的明确适配，无测试阈值调整。

v3实际仍有6/24契约失败，说明服务**接受但未保证执行**JSON Schema；前述探测仅证明单次成功。v4简化为扁平事件契约（kind/start/end/status/evidence），不让主检测输出内部阶段，不丢弃/修补多余字段。原机制真值可保留阶段，本轮不以全部为空的模型阶段输出报阶段成绩。后续阶段研究须独立登记、先确认活动。三批开发失败保留，正式测试不调提示。

六类：normal、spike、step、transient、deformation、mixed。简单白噪声的尺度按站点与分量抽样，正常线性趋势和年度季节项，小比例独立缺测和短中断。持续运动由速度积分，依次加速、持续、减速；停止后的平台不延长活动标签。mixed在活动内部叠加三类质量形态，支持共存标签。不叠加复杂水文、闪烁噪声或站网；噪声状态变化和复杂季节异常不在本轮覆盖范围。

机制标签用于训练与评价，**不是观测专家金标准**，弱形变和边界仍有可识别性限制。论文需要补独立专家复核及真实站点验证。观测文件仅匿名站点、日期、NEU和历史背景范围；场景、参数与潜在轨迹单独存在truth目录。数值监督仅读训练站truth；视觉输入不读truth。登记冻结源码、协议、依赖版本、输入哈希、方法、模型、预算与并发。评分仅评价正常前缀之后；缺测不赋正常。

## 方法与原代码标注

| 方法 | 信息/实现 | 原版、适配与参数 |
|---|---|---|
| random_forest | sklearn RF，多标签GNSS局部特征 | 原始库默认参数/决策，固定random_state；不是Crocetti作者代码复现 |
| rf_balanced | 同特征RF，稀有类别学习对照 | **特别调整**class_weight=balanced，另列结果；决策规则和其余参数不变 |
| extra_trees | sklearn ExtraTrees，同训练与特征 | 同上，强同信息学习对照 |
| isolation_forest | sklearn iForest，仅正常历史训练 | 默认contamination=auto与库判定；只能评价通用异常 |
| visual_only | 原始NEU图、正常历史范围、缺测 | qwen3.8-flash，temperature=0，enable_thinking=False |
| fusion | 同图＋背景尺度/相对残差/21日局部统计 | 本项目单轮方法，无统一毫米或速率阈值 |
| fusion_no_reference | 原始图＋原始21日块中位点 | 移除历史范围与其派生尺度/残差；背景信息组消融，不是单特征因果效应 |
| fusion_n_only | fusion仅保留N图和N证据 | 三分量信息消融 |
| visualtime_gnss | **实际调用作者原始**prompt.create_openai_request/create_vision_messages | GNSS绘图、保留索引/缺测、按NEU分别point/range、union评分、模型替换、严格解析及不重试均标为适配；原temperature=0.4与stop不变；无阈值调整 |

fusion去掉数值证据就是visual_only，不能重复当成另一独立方法。21日居中统计包含未来数据，只作离线辅助，不能以提前响应扩展原始活动。历史前缀拟合线性＋年周期背景，以残差MAD估尺度；不设人为毫米尺度下限。数值特征缺口在分类器内用训练侧中位数处理，图和日期不插补。统计证据不是已校准检验概率。Library baseline不是作者论文源码；不能冒称完整论文成绩复现。

VisualTimeAnomaly原版源码锁定commit且保持干净，原始函数/默认值/适配列表分别存在source.json、adaptation.json、request.json和response.json。当前公开版本未提供完整TSAD-Agents，不能自写后称原版。VLM4TS源码也已锁定，其默认需要CLIP权重与PyTorch环境；未运行则不填成绩。其去趋势、阈值和模型版本局限见文献笔记。

## 顺序与预算

1. 本地契约测试与数值小批。
2. 验证6站×4视觉处理=24真实请求；确认解析、索引、缺测和阶段契约，再扩量。保留全部失败，不因成绩挑样本。
3. 作者源码GNSS小批：6站×3轴×2任务=36请求。小批合计60次调用。
4. 新种子主批：训练、验证、测试各24站，测试每场景4站；四数值、四视觉、作者源码适配共同比较。主测试视觉96＋作者源码144＝240请求。
5. 主批fusion与visual_only第二遍独立调用48请求，RF确定性重放，描述模型响应波动；重复推理不是新独立站点。

并发≤4、默认3；单请求120秒网络超时，禁止隐式重试。用户已授权先小批再大批且不限制总请求，但接口失败时先查证再新登记。若开发发现接口缺陷，记录修订，不修改已冻结测试追求高分。效果不高于对照也保留结果。

```powershell
uv run gnss-research generate --out data/main --seed 2026101202 --per-split 24
uv run gnss-research register --dataset data/main --out runs/main --split test --methods random_forest rf_balanced extra_trees isolation_forest visual_only fusion fusion_no_reference fusion_n_only visualtime_gnss --model qwen3.8-flash --max-requests 240
uv run gnss-research run --run runs/main --env-file .env
```

## 终点

主终点为deformation逐日P/R/F1和一对一事件F1（IoU≥0.5），IoU≥0.3补充。spike/step按发生日期严格点匹配，±3日仅补充；transient按区间。本轮只定位持续形变活动，不预测内部阶段。内部真值的阶段诊断不是本轮模型终点，空阶段的零分不作为方法比较；未来阶段实验必须先确认活动且限定内部。

作者通用异常源码/iForest与其余方法比较通用异常union，不硬转为形变。数值方法有连续分数才报Average Precision（AP）；视觉二值输出不伪造概率。无异常站误报事件/站年单列，全部真实阴性日的报警负担另列。缺测排除但报告天数；待定和调用失败仍在已知阳性上记漏检，并单列失败、未执行和待定数。无正样本站F1记0，解释时区分station macro与micro。

不做point adjustment、不在测试取F1-max或扫描阈值。没有统一物理阈值不等于模型没有统计决策边界。未来调整阈值、窗口或后处理必须训练/验证侧登记，另列tuned方法并保留原默认结果。

整站成对bootstrap 2000次，给F1差值和95%百分位区间；通用异常与可用方法的形变分别比较。小站点数、同一生成器和多消融意味着不宣称现场总体显著性，不把重复调用当独立样本。运行时间、真实调用数、实际token/模型单独记，未有价格凭据不编造货币成本。后续实验前持续更新来源核查，本次定向调研不冒称系统综述。
